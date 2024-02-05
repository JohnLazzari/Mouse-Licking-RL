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
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name
from sac_model import Actor, Critic, Actor_Seq
from algorithms import sac, select_action, hard_update, REINFORCE

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

def sac_learn(
    env,
    seed,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    critic,
    actor_optimizer_spec,
    critic_optimizer_spec,
    replay_buffer_size,
    batch_size,
    alpha,
    gamma,
    automatic_entropy_tuning, 
    learning_starts,
    learning_freq,
    save_iter,
    log_steps,
    frame_skips,
    model_save_path,
    reward_save_path,
    steps_save_path,
    model
):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Box

    actor_bg = actor(inp_dim, hid_dim, action_dim).cuda()
    critic_bg = critic(action_dim+inp_dim, hid_dim).cuda()
    critic_target_bg = critic(action_dim+inp_dim, hid_dim).cuda()
    hard_update(critic_target_bg, critic_bg)

    param_names = []
    for name, param in actor_bg.named_parameters():
        param_names.append(name)
    
    if model == "rnn_seq":
        actor_bg_optimizer = actor_optimizer_spec.constructor(actor_bg.parameters(), names=param_names, **actor_optimizer_spec.kwargs)
    elif model == "rnn":
        actor_bg_optimizer = actor_optimizer_spec.constructor(actor_bg.parameters(), **actor_optimizer_spec.kwargs)

    critic_bg_optimizer = critic_optimizer_spec.constructor(critic_bg.parameters(), **critic_optimizer_spec.kwargs)

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
    total_episodes = 0
    avg_reward = [0]
    avg_steps = [0]

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset(0)
    ep_trajectory = []

    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
    if model == "rnn_seq":
        actor_bg.reset_y(1)

    # TODO learning for new network isnt correct, y is not updated during the processing of the sequence, need to make a custom rnn
    # need to make sure I update params every step as opposed to after every episode (too much bad experience)
    # organize code such that its easy to switch and add things between the standard rnn and new rnn
    ### STEPS PER EPISODE ###
    for t in count():

        with torch.no_grad():
            action, h_current = select_action(actor_bg, state, h_prev, evaluate=False)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        for _ in range(frame_skips):
            next_state, reward, done = env.step(episode_steps, action, h_prev)
            episode_steps += 1
            episode_reward += reward
            if done == True:
                break

        mask = 1 if episode_steps == env.max_timesteps else float(not done)

        ep_trajectory.append((state, action, reward, next_state, mask))

        state = next_state
        h_prev = h_current

        ### EARLY TERMINATION OF EPISODE
        if done:

            total_episodes += 1

            # Add stats to lists
            avg_steps.append(episode_steps)
            avg_reward.append(episode_reward)

            # Push the episode to replay
            policy_memory.push(ep_trajectory)

            # reset training conditions
            h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
            state = env.reset(total_episodes) 

            # resest lists
            ep_trajectory = []

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

            print("Episode %d" % (total_episodes,))
            print("mean reward (10 episodes): %f" % episode_reward)
            print("mean steps (10 episodes): %f" % episode_steps)
            print("best mean reward: %f" % best_mean_episode_reward)
            sys.stdout.flush()

            if total_episodes % log_steps == 0:
                # Dump statistics to pickle
                np.save(f'{reward_save_path}.npy', Statistics["mean_episode_rewards"])
                np.save(f'{steps_save_path}.npy', Statistics["mean_episode_steps"])
                print("Saved to %s" % 'training_reports')
                print('--------------------------\n')
            
            if total_episodes % save_iter == 0:
                torch.save({
                    'iteration': t,
                    'agent_state_dict': actor_bg.state_dict(),
                    'critic_state_dict': critic_bg.state_dict(),
                    'critic_target_state_dict': critic_target_bg.state_dict(),
                    'agent_optimizer_state_dict': actor_bg_optimizer.state_dict(),
                    'critic_optimizer_state_dict': critic_bg_optimizer.state_dict(),
                }, model_save_path + str(total_episodes) + '.pth')

            # reset tracking variables
            episode_steps = 0
            episode_reward = 0

            # Apply Basal Ganglia update (using SAC)
            if len(policy_memory.buffer) > batch_size and total_episodes > learning_starts and total_episodes % learning_freq == 0:

                sac(actor_bg,
                    critic_bg,
                    critic_target_bg,
                    critic_bg_optimizer,
                    actor_bg_optimizer,
                    policy_memory,
                    batch_size,
                    hid_dim,
                    gamma,
                    automatic_entropy_tuning,
                    log_alpha,
                    target_entropy,
                    alpha,
                    alpha_optim,
                    model)

            if model == "rnn_seq":
                actor_bg.reset_y(1)
        


    