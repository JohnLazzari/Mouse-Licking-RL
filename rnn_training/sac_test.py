import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
import scipy.io
from lick_env import Lick_Env_Cont, Kinematics_Jaw_Env
import matplotlib.pyplot as plt
import matplotlib
from algorithms import select_action

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

from utils.gym import get_wrapper_by_name
from sac_model import Actor, Critic
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d


USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

ENV = "kinematics_jaw"
INP_DIM = 11
HID_DIM = 256
ACTION_DIM = 2
THRESH = 1
ACT_SCALE = 1
ACT_BIAS = 0
DT = 0.01
TIMESTEPS = int(3 / DT)
CHECK_PATH = "checkpoints/kinematics_jaw.pth"
SAVE_PATH = "results/test_activity/kinematics_jaw_act.npy"
BETA = .99
BG_SCALE = .04
FRAMESKIP = 2
ALM_DATA = "data/PCs_PSTH"
KINEMATICS_DATA = "data/kinematics"

def test(
    env,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    critic,
    check_path,
    save_path,
    frameskips,
    act_scale,
    act_bias
):
    checkpoint = torch.load(check_path)

    _actor = actor(inp_dim, hid_dim, action_dim, act_scale, act_bias).cuda()
    _actor.load_state_dict(checkpoint['agent_state_dict'])

    episode_reward = 0
    episode_steps = 0
    str_activity = {}

    ### GET INITAL STATE + RESET MODEL BY POSE
    state = env.reset(0)
    #num_layers specified in the policy model 
    h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
    
    str_activity[0] = []
    str_activity[1] = []
    str_activity[2] = []

    ### STEPS PER EPISODE ###
    for conditions in range(3):

        for t in count():

            with torch.no_grad():
                action, h_current = select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(frameskips):
                str_activity[env.cur_cond].append(h_prev.squeeze().cpu().numpy())
                next_state, reward, done = env.step(episode_steps, action, h_prev, conditions)
                episode_steps += 1
                episode_reward += reward
                if done:
                    break

            state = next_state
            h_prev = h_current

            if done:
                h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
                state = env.reset(conditions+1) 
                episode_reward = 0
                episode_steps = 0
                break

    # reset tracking variables
    print(np.array(str_activity[0]).shape)
    A_agent = gaussian_filter1d(np.array(str_activity[0]), 2, axis=0)
    psth = np.mean(A_agent, axis=-1)
    plt.plot(psth)
    plt.show()

    switch_0_pca = PCA(n_components=3)
    switch_0_projected = switch_0_pca.fit_transform(A_agent)

    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(switch_0_projected[:, 0], switch_0_projected[:, 1], switch_0_projected[:, 2])
    plt.show()

    # plot trajectories
    plt.plot(switch_0_projected[:, 0], label="str pc1 1s")
    plt.plot(switch_0_projected[:, 1], label="str pc2 1s")
    plt.plot(switch_0_projected[:, 2], label="str pc3 1s")
    plt.axvline(100, linestyle='dashed')
    plt.legend()
    plt.show()

    np.save(save_path, np.array([str_activity[0], str_activity[1], str_activity[2]]))

if __name__ == "__main__":

    if ENV == "kinematics_jaw":
        env = Kinematics_Jaw_Env(ACTION_DIM, DT, KINEMATICS_DATA, ALM_DATA)
    elif ENV == "lick_ramp":
        env = Lick_Env_Cont(ACTION_DIM, TIMESTEPS, THRESH, DT, BETA, BG_SCALE, ALM_DATA)
        
    test(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, Critic, CHECK_PATH, SAVE_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS)
