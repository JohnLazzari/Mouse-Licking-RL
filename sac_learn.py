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
from sac_model import Actor, Critic
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
    optimizer_spec,
    replay_buffer_size=1000000,
    batch_size=8,
    alpha=0.20,
    gamma=0.99,
    automatic_entropy_tuning=False, 
    learning_starts=50000,
    learning_freq=1,
    save_iter=100000,
    model_save_path="checkpoints/cont_lick_check",
    training_save_folder="training_reports"
):

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Box

    actor_bg = actor(inp_dim, hid_dim, action_dim).cuda()
    critic_bg = critic(action_dim+inp_dim, hid_dim).cuda()
    critic_target_bg = critic(action_dim+inp_dim, hid_dim).cuda()
    hard_update(critic_target_bg, critic_bg)
    
    actor_bg_optimizer = optimizer_spec.constructor(actor_bg.parameters(), **optimizer_spec.kwargs)
    critic_bg_optimizer = optimizer_spec.constructor(critic_bg.parameters(), **optimizer_spec.kwargs)

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
    LOG_EVERY_N_STEPS = 10

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset(0)
    ep_trajectory = []

    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")

    ### STEPS PER EPISODE ###
    for t in count():

        with torch.no_grad():
            action, h_current = select_action(actor_bg, state, h_prev, evaluate=False)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = env.step(episode_steps, action)
        episode_reward += reward
        episode_steps += 1

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

            if total_episodes % LOG_EVERY_N_STEPS == 0:
                # Dump statistics to pickle
                np.save(f'{training_save_folder}/rewards.npy', Statistics["mean_episode_rewards"])
                np.save(f'{training_save_folder}/steps.npy', Statistics["mean_episode_steps"])
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
                    'alm_network_state_dict': env.alm.state_dict(),
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
                alpha_optim)
        


    