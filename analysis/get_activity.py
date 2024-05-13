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

INP_DIM = 6
HID_DIM = 64
ACTION_DIM = 1
THRESH = 1
ACT_SCALE = 0.5
ACT_BIAS = 0.5
DT = 0.01
TIMESTEPS = int(3 / DT)
BETA = .99
BG_SCALE = 0.1
FRAMESKIP = 1
NUM_CONDITIONS = 1
ALM_HID_UNITS = 4
TRAJECTORY = False

ENV = "lick_ramp"
CHECK_PATH = "checkpoints/lick_attractor_lowd.pth"
ALM_NET_PATH = "checkpoints/rnn_goal_data_delay.pth"
SAVE_PATH = "results/test_activity/lick_attractor_lowd_act.npy"

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
    num_conditions,
    alm_hid_units,
):

    checkpoint = torch.load(check_path)
    _actor = actor(inp_dim, hid_dim, action_dim, act_scale, act_bias).cuda()
    _actor.load_state_dict(checkpoint['agent_state_dict'])

    episode_reward = 0
    episode_steps = 0
    str_activity = {}
    str_output = {}

    ### STEPS PER EPISODE ###
    for conditions in range(num_conditions):

        h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
        state = env.reset(conditions) 
        episode_reward = 0
        episode_steps = 0
        str_activity[conditions] = []
        str_output[conditions] = []

        for t in range(int(env.max_timesteps / frameskips)):

            with torch.no_grad():
                action, h_current = select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(frameskips):
                str_activity[conditions].append(h_prev.squeeze().cpu().numpy())
                str_output[conditions].append(action)
                next_state, reward, done = env.step(episode_steps, action, conditions)
                episode_steps += 1
                episode_reward += reward
                if done:
                    break

            state = next_state
            h_prev = h_current

            if done:
                break

    # Print STR PSTH
    A_agent = gaussian_filter1d(np.array(str_activity[0]), 2, axis=0)
    psth = np.mean(A_agent, axis=-1)
    plt.plot(psth)
    plt.show()

    # Print the output layer
    plt.plot(str_output[0])
    plt.show()

    # Do PCA on STR activity
    switch_0_pca = PCA(n_components=3)
    switch_0_projected = switch_0_pca.fit_transform(A_agent)

    # Show 3D trajectory
    ax = plt.figure().add_subplot(projection='3d')
    ax.plot(switch_0_projected[:, 0], switch_0_projected[:, 1], switch_0_projected[:, 2])
    plt.show()

    # plot trajectories over each other
    plt.plot(switch_0_projected[:, 0], label="str pc1 1s")
    plt.plot(switch_0_projected[:, 1], label="str pc2 1s")
    plt.plot(switch_0_projected[:, 2], label="str pc3 1s")
    plt.axvline(1, linestyle='dashed')
    plt.legend()
    plt.show()

    # Save the activity for further testing
    np.save(save_path, str_activity[0])

if __name__ == "__main__":

    env = Lick_Env_Cont(ACTION_DIM, 
                        TIMESTEPS, 
                        THRESH, 
                        DT, 
                        BETA, 
                        BG_SCALE, 
                        TRAJECTORY, 
                        ALM_NET_PATH, 
                        ALM_HID_UNITS)

    test(env, 
         INP_DIM, 
         HID_DIM, 
         ACTION_DIM, 
         Actor, 
         CHECK_PATH, 
         SAVE_PATH, 
         FRAMESKIP, 
         ACT_SCALE, 
         ACT_BIAS, 
         NUM_CONDITIONS, 
         ALM_HID_UNITS)
