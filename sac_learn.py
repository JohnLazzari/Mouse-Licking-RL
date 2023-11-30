import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name
from sac_model import Actor, Critic

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

def select_action(policy: Actor, state: list, hn: torch.Tensor, evaluate: bool) -> (list, torch.Tensor):
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0).cuda()
    hn = hn.cuda()

    if evaluate == False: 
        action, _, _, hn, _ = policy.sample(state, hn, sampling=True)
    else:
        _, _, action, hn, _ = policy.sample(state, hn, sampling=True)

    return action.detach().cpu().tolist()[0], hn.detach()

def soft_update(target: Critic, source: Critic, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target: Critic, source: Critic):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def sac_learn(
    env,
    seed,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    critic,
    optimizer_spec,
    replay_buffer_size=1000000,
    batch_size=8,
    alpha=0.20,
    gamma=0.99,
    automatic_entropy_tuning=False, 
    learning_starts=50000,
    learning_freq=1,
    save_iter=100000,
    save_path="checkpoints/cont_lick_check",
):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Box

    _actor = actor(inp_dim, hid_dim, action_dim).cuda()
    _critic = critic(action_dim+inp_dim, hid_dim).cuda()
    _critic_target = critic(action_dim+inp_dim, hid_dim).cuda()
    hard_update(_critic_target, _critic)
    
    actor_optimizer = optimizer_spec.constructor(_actor.parameters(), **optimizer_spec.kwargs)
    critic_optimizer = optimizer_spec.constructor(_critic.parameters(), **optimizer_spec.kwargs)

    if automatic_entropy_tuning is True:
        target_entropy = -env.action_space.shape[0]
        log_alpha = torch.zeros(1, requires_grad=True, device="cuda:0")
        alpha_optim = optim.Adam([log_alpha], lr=.0003)

    policy_memory = ReplayBuffer(replay_buffer_size, seed)

    Statistics = {
        "mean_episode_rewards": [],
        "mean_episode_steps": [],
        "best_mean_episode_rewards": []
    }

    episode_reward = 0
    best_mean_episode_reward = -float("inf")
    episode_steps = 0
    avg_reward = [0]
    avg_steps = [0]
    LOG_EVERY_N_STEPS = 10

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset()
    ep_trajectory = []

    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim))

    ### STEPS PER EPISODE ###
    for t in count():

        # slightly bring the threshold down during training
        #if t % 250_000 == 0 and env.thresh > .005:
        #    env.thresh -= .001

        with torch.no_grad():
            action, h_current = select_action(_actor, state, h_prev, evaluate=False)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = env.step(episode_steps%env.max_timesteps, action)
        episode_reward += reward
        episode_steps += 1

        mask = 1 if episode_steps == env.max_timesteps else float(not done)

        ep_trajectory.append((state, action, reward, next_state, mask))

        state = next_state
        h_prev = h_current

        ### EARLY TERMINATION OF EPISODE
        if done:
            # Add stats to lists
            avg_steps.append(episode_steps)
            avg_reward.append(episode_reward)
            # Push the episode to replay
            policy_memory.push(ep_trajectory)
            # reset training conditions
            h_prev = torch.zeros(size=(1, 1, hid_dim))
            state = env.reset()
            ep_trajectory = []
            # reset tracking variables
            episode_steps = 0
            episode_reward = 0

        if len(policy_memory.buffer) > batch_size and t > learning_starts and t % learning_freq == 0:
            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = policy_memory.sample(batch_size)

            state_batch = pad_sequence(state_batch, batch_first=True).cuda()
            action_batch = pad_sequence(action_batch, batch_first=True).cuda()
            reward_batch = pad_sequence(reward_batch, batch_first=True).unsqueeze(-1).cuda()
            next_state_batch = pad_sequence(next_state_batch, batch_first=True).cuda()
            mask_batch = pad_sequence(mask_batch, batch_first=True).unsqueeze(-1).cuda()

            h_train = torch.zeros(size=(1, batch_size, hid_dim))
            with torch.no_grad():
                next_state_action, next_state_log_pi, _, _, _ = _actor.sample(next_state_batch, h_train, sampling=False)
                qf1_next_target, qf2_next_target = _critic_target(next_state_batch, next_state_action, h_train)
                min_qf_next_target = torch.minimum(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = reward_batch + mask_batch * gamma * (min_qf_next_target)

            qf1, qf2 = _critic(state_batch, action_batch, h_train)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            critic_optimizer.zero_grad()
            qf_loss.backward()
            critic_optimizer.step()

            pi_action_bat, log_prob_bat, _, _, _ = _actor.sample(state_batch, h_train, sampling= False)

            qf1_pi, qf2_pi = _critic(state_batch, pi_action_bat, h_train)
            min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

            policy_loss = ((alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

            actor_optimizer.zero_grad()
            policy_loss.backward()
            actor_optimizer.step()

            if automatic_entropy_tuning:
                alpha_loss = -(log_alpha * (log_prob_bat + target_entropy).detach()).mean()

                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()

                alpha = log_alpha.exp()

            soft_update(_critic_target, _critic, .005)
            h_train = h_train.detach()
        
        ### 4. Log progress and keep track of statistics
        if len(avg_reward) > 0:
            mean_episode_reward = np.mean(np.array(avg_reward)[-10:])
        if len(avg_steps) > 0:
            mean_episode_steps = np.mean(np.array(avg_steps)[-10:])
        if len(avg_reward) > 10:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

        Statistics["mean_episode_rewards"].append(mean_episode_reward)
        Statistics["mean_episode_steps"].append(mean_episode_steps)
        Statistics["best_mean_episode_rewards"].append(best_mean_episode_reward)

        if t % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
            print("Timestep %d" % (t,))
            print("mean reward (100 episodes): %f" % mean_episode_reward)
            print("mean steps (100 episodes): %f" % mean_episode_steps)
            print("best mean reward: %f" % best_mean_episode_reward)
            sys.stdout.flush()

            # Dump statistics to pickle
            with open('statistics.pkl', 'wb') as f:
                pickle.dump(Statistics, f)
                print("Saved to %s" % 'statistics.pkl')
                print('--------------------------\n')
        
        if t % save_iter == 0 and t > learning_starts:
            torch.save({
                'iteration': t,
                'agent_state_dict': _actor.state_dict(),
                'critic_state_dict': _critic.state_dict(),
                'critic_target_state_dict': _critic_target.state_dict(),
                'agent_optimizer_state_dict': actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': critic_optimizer.state_dict(),
            }, save_path + str(t) + '.pth')


    