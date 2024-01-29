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
from algorithms import select_action

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name
from sac_model import Actor, Critic
from sklearn.decomposition import PCA

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

INP_DIM = 3
HID_DIM = 32
ACTION_DIM = 1
THRESH = 1
DT = 0.1
TIMESTEPS = int(6 / DT)
CHECK_PATH = "checkpoints/cont_lick_check20000.pth"
SAVE_PATH = "results"


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
    alm_activity = {}
    str_activity = {}

    ### STEPS PER EPISODE ###
    for conditions in range(2):

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = env.reset(conditions)
        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
        
        alm_activity[env.switch] = []
        str_activity[env.switch] = []

        for t in range(env.max_timesteps):

            alm_activity[env.switch].append(state[0])
            str_activity[env.switch].append(h_prev.squeeze().cpu().numpy())

            with torch.no_grad():
                action, h_current = select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            next_state, reward, done = env.step(t, action)

            state = next_state
            h_prev = h_current

            if done:
                break

    # reset tracking variables
    switch_0_pca = PCA(n_components=3)
    switch_0_projected = switch_0_pca.fit_transform(np.array(str_activity[0]))

    switch_1_pca = PCA(n_components=3)
    switch_1_projected = switch_1_pca.fit_transform(np.array(str_activity[1]))

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(switch_0_projected[:, 0], switch_0_projected[:, 1], switch_0_projected[:, 2])
    ax.plot(switch_1_projected[:, 0], switch_1_projected[:, 1], switch_1_projected[:, 2])
    plt.show()

    # plot trajectories
    plt.plot(switch_0_projected[:, 0], label="str pc1 3s")
    plt.plot(switch_1_projected[:, 0], label="str pc1 1s")
    plt.plot(alm_activity[0], label="alm pc1 3s")
    plt.plot(alm_activity[1], label="alm pc1 1s")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    env = Lick_Env_Cont(ACTION_DIM, TIMESTEPS, THRESH, DT)
    test(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, Critic, CHECK_PATH, SAVE_PATH)
