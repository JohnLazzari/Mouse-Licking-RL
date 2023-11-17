import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import scipy.io
from lick_env import Lick_Env_Cont
import matplotlib.pyplot as plt

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

alm_activity = scipy.io.loadmat("alm_warped_activity_3pcs_1slick.mat")
alm_activity_arr = alm_activity["warped_activity_3pcs_1slick"]

INP_DIM = 6+64
HID_DIM = 256
ACTION_DIM = 8
TARGET_DYNAMICS = alm_activity_arr
THRESH = 0.16
ALM_HID = 64
CHECK_PATH = "checkpoints/cont_lick_check1300000.pth"
SAVE_PATH = "learned_trajectories/trajectory.npy"

def select_action(policy: Actor, state: list, hn: torch.Tensor, evaluate: bool) -> (list, torch.Tensor):
    state = torch.tensor(state).unsqueeze(0).unsqueeze(0).cuda()
    hn = hn.cuda()

    if evaluate == False: 
        action, _, _, hn, _ = policy.sample(state, hn, sampling=True)
    else:
        _, _, action, hn, _ = policy.sample(state, hn, sampling=True)

    return action.detach().cpu().tolist()[0], hn.detach()

def test(
    env,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    critic,
    check_path,
    save_path
):
    checkpoint = torch.load(check_path)

    _actor = actor(inp_dim, hid_dim, action_dim).cuda()
    _critic = critic(action_dim+inp_dim, hid_dim).cuda()
    _critic_target = critic(action_dim+inp_dim, hid_dim).cuda()

    _actor.load_state_dict(checkpoint['agent_state_dict'])
    _critic.load_state_dict(checkpoint['critic_state_dict'])
    _critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
    
    episode_reward = 0
    episode_steps = 0
    LOG_EVERY_N_STEPS = 1000
    trajectory = []
    actor_trajectory = []

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset()
    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim))

    ### STEPS PER EPISODE ###
    for t in count():

        trajectory.append(state[-3:])
        actor_trajectory.append(state[-6:-3])

        # slightly bring the threshold down during training
        if t % 150_000 == 0 and env.thresh > .01:
            env.thresh -= .01

        with torch.no_grad():
            action, h_current =  select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

        ### TRACKING REWARD + EXPERIENCE TUPLE###
        next_state, reward, done = env.step(episode_steps%env.max_timesteps, action)
        episode_reward += reward
        episode_steps += 1

        state = next_state
        h_prev = h_current

        ### EARLY TERMINATION OF EPISODE
        if done:
            # reset training conditions
            h_prev = torch.zeros(size=(1, 1, hid_dim))
            state = env.reset()
            # reset tracking variables
            episode_steps = 0
            episode_reward = 0
            trajectory = np.array(trajectory)
            actor_trajectory = np.array(actor_trajectory)
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(trajectory[:,0], trajectory[:,1], trajectory[:,2])
            ax.scatter(actor_trajectory[:,0], actor_trajectory[:,1], actor_trajectory[:,2])
            plt.show()
            trajectory = []
            actor_trajectory = []
            np.save(save_path, np.array(trajectory))

if __name__ == "__main__":
    env = Lick_Env_Cont(ACTION_DIM, TARGET_DYNAMICS, THRESH, ALM_HID)
    test(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, Critic, CHECK_PATH, SAVE_PATH)
