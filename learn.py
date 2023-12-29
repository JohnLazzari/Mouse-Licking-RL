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
from algorithms import sac, select_action_ac, hard_update, REINFORCE, One_Step_AC

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
    save_path="checkpoints/cont_lick_check",
):
    ### Switching to one-step actor critic (Goodbye SAC) ###

    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Box

    actor_bg = actor(inp_dim, hid_dim, action_dim).cuda()
    critic_bg = critic(inp_dim, hid_dim).cuda()
    
    actor_bg_optimizer = optimizer_spec.constructor(actor_bg.parameters(), **optimizer_spec.kwargs)
    critic_bg_optimizer = optimizer_spec.constructor(critic_bg.parameters(), **optimizer_spec.kwargs)

    Statistics = {
        "mean_episode_rewards": [],
        "mean_episode_steps": [],
        "best_mean_episode_rewards": []
    }

    episode_reward = 0
    best_mean_episode_reward = -float("inf")
    episode_steps = 0
    total_episodes = 0
    avg_reward = []
    avg_steps = []
    LOG_EVERY_N_STEPS = 1

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset(0)
    ep_trajectory = []

    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")

    # I
    I = 1

    ### STEPS PER EPISODE ###
    for t in count():

        with torch.no_grad():
            action, h_next, = select_action_ac(actor_bg, state, h_prev)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = env.step(episode_steps%env.max_timesteps, action)

        mask = 1 if episode_steps == env.max_timesteps else float(not done)

        if episode_steps == 0:
            tuple = (state, action, reward, next_state, mask, h_prev, h_next)
        else:
            tuple = (state, action, reward, next_state, mask)

        ep_trajectory.append(tuple)

        I = One_Step_AC(ep_trajectory, actor_bg, critic_bg, actor_bg_optimizer, critic_bg_optimizer, gamma, I)

        state = next_state
        h_prev = h_next

        episode_reward += reward
        episode_steps += 1

        ### EARLY TERMINATION OF EPISODE
        if done:

            # I
            I = 1

            total_episodes += 1

            # Add stats to lists
            avg_steps.append(episode_steps)
            avg_reward.append(episode_reward)

            # reset training conditions
            h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
            state = env.reset(total_episodes)

            # resest lists
            ep_trajectory = []

            # reset tracking variables
            episode_steps = 0
            episode_reward = 0

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

            if total_episodes % LOG_EVERY_N_STEPS == 0 and t > learning_starts:
                print("Timestep %d" % (t,))
                print("mean reward (10 episodes): %f" % mean_episode_reward)
                print("mean steps (10 episodes): %f" % mean_episode_steps)
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
                    'agent_state_dict': actor_bg.state_dict(),
                    'critic_state_dict': critic_bg.state_dict(),
                    'agent_optimizer_state_dict': actor_bg_optimizer.state_dict(),
                    'critic_optimizer_state_dict': critic_bg_optimizer.state_dict(),
                    'alm_network_state_dict': env.alm.state_dict(),
                }, save_path + str(t) + '.pth')