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
HID_DIM = 256
ACTION_DIM = 1
THRESH = 1
DT = 0.01
TIMESTEPS = int(3 / DT)
CHECK_PATH = "checkpoints/lick_ramp30000.pth"
SAVE_PATH = "results/lick_ramp30000_fr.npy"
BETA = .99
BG_SCALE = .05
FRAMESKIP = 2
ALM_DATA = "data/alm_fr_averaged_1s.mat"

def test(
    env,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    critic,
    check_path,
    save_path,
    frameskips
):
    checkpoint = torch.load(check_path)

    _actor = actor(inp_dim, hid_dim, action_dim).cuda()
    _actor.load_state_dict(checkpoint['agent_state_dict'])

    episode_reward = 0
    episode_steps = 0
    alm_activity = {}
    str_activity = {}

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset(0)
    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
    
    alm_activity[env.switch] = []
    str_activity[env.switch] = []

    ### STEPS PER EPISODE ###
    for conditions in range(1):

        for t in count():

            with torch.no_grad():
                action, h_current = select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(frameskips):
                alm_activity[env.switch].append(state[0])
                str_activity[env.switch].append(h_prev.squeeze().cpu().numpy())
                next_state, reward, done = env.step(episode_steps, action, h_prev)
                episode_steps += 1
                episode_reward += reward
                if done:
                    break

            state = next_state
            h_prev = h_current

            if done:
                break

    # reset tracking variables
    switch_0_pca = PCA(n_components=3)
    switch_0_projected = switch_0_pca.fit_transform(np.array(str_activity[1]))

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(switch_0_projected[:, 0], switch_0_projected[:, 1], switch_0_projected[:, 2])
    plt.show()

    # plot trajectories
    plt.plot(switch_0_projected[:, 0], label="str pc1 3s")
    plt.plot(alm_activity[1], label="alm pc1 3s")
    plt.legend()
    plt.show()

    np.save(save_path, np.array(str_activity[1]))

if __name__ == "__main__":
    env = Lick_Env_Cont(ACTION_DIM, TIMESTEPS, THRESH, DT, BETA, BG_SCALE, ALM_DATA)
    test(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, Critic, CHECK_PATH, SAVE_PATH, FRAMESKIP)
