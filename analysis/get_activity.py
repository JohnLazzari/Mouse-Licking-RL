import sys
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from pathlib import Path

HOME = str(Path.home())
CONFIG_PATH = os.path.join(HOME, 'Mouse-Licking-RL/rnn_training')
sys.path.insert(0, CONFIG_PATH)

from sac_model import Actor
from lick_env import Lick_Env_Cont, Kinematics_Jaw_Env
from algorithms import select_action

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

ENV = "lick_ramp"
INP_DIM = 4
HID_DIM = 256
ACTION_DIM = 1
THRESH = 1
ACT_SCALE = 0.5
ACT_BIAS = 0.5
DT = 0.01
TIMESTEPS = int(3 / DT)
CHECK_PATH = "checkpoints/lick_attractor.pth"
SAVE_PATH = "results/test_activity/lick_attractor_act.npy"
BETA = .99
BG_SCALE = 0.1
FRAMESKIP = 2
NUM_CONDITIONS = 1
ALM_DATA = "data/PCs_PSTH"
KINEMATICS_DATA = "data/kinematics"

def test(
    env,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    check_path,
    save_path,
    frameskips,
    act_scale,
    act_bias,
    num_conditions
):

    checkpoint = torch.load(check_path)
    _actor = actor(inp_dim, hid_dim, action_dim, act_scale, act_bias).cuda()
    _actor.load_state_dict(checkpoint['agent_state_dict'])

    episode_reward = 0
    episode_steps = 0
    str_activity = {}

    ### STEPS PER EPISODE ###
    for conditions in range(num_conditions):

        h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
        state = env.reset(conditions) 
        episode_reward = 0
        episode_steps = 0
        str_activity[conditions] = []

        for t in range(int(env.max_timesteps / frameskips)):

            with torch.no_grad():
                action, h_current = select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(frameskips):
                str_activity[conditions].append(h_prev.squeeze().cpu().numpy())
                next_state, reward, done = env.step(episode_steps, action, conditions)
                episode_steps += 1
                episode_reward += reward
                if done:
                    break

            state = next_state
            h_prev = h_current

            if done:
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

    np.save(save_path, str_activity[0])

if __name__ == "__main__":

    if ENV == "kinematics_jaw":
        env = Kinematics_Jaw_Env(ACTION_DIM, DT, KINEMATICS_DATA, ALM_DATA, BG_SCALE)
    elif ENV == "lick_ramp":
        env = Lick_Env_Cont(ACTION_DIM, TIMESTEPS, THRESH, DT, BETA, BG_SCALE, ALM_DATA)
        
    test(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, SAVE_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS)
