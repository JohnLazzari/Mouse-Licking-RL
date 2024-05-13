import sys
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl

HOME = str(Path.home())
CONFIG_PATH = os.path.join(HOME, 'Mouse-Licking-RL/rnn_training')
sys.path.insert(0, CONFIG_PATH)

from sac_model import Actor
from lick_env import Lick_Env_Cont, Kinematics_Jaw_Env
from algorithms import select_action

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
plt.rcParams.update({'font.size': 24})

def NormalizeData(data, min, max):
    return (data - min) / (max - min)

ENV = "lick_ramp"
INP_DIM = 6
HID_DIM = 64
ACTION_DIM = 1
THRESH = 1
ACT_SCALE = 0.5
ACT_BIAS = 0.5
DT = 0.01
TIMESTEPS = int(3 / DT)
CHECK_PATH = "checkpoints/lick_attractor_lowd.pth"
ALM_NET_PATH = "checkpoints/rnn_goal_data_delay.pth"
BETA = .99
BG_SCALE = 0.1
FRAMESKIP = 1
NUM_CONDITIONS = 1
ALM_HID_UNITS = 4
TRAJECTORY = False

def get_ramp_mode(A_agent, A_alm):
    
    # use ramp mode instead of psth (choose baseline)
    baseline_str = np.zeros(A_agent.shape[1])
    baseline_alm = np.zeros(A_alm.shape[1])

    # ALM Ramp Mode
    mean_peak = np.mean(A_alm[100:110], axis=0)
    ramp_mode_alm = mean_peak - baseline_alm
    ramp_mode_alm /= np.linalg.norm(ramp_mode_alm)

    # STR Ramp Mode
    mean_peak = np.mean(A_agent[100:110], axis=0)
    ramp_mode_str = mean_peak - baseline_str
    ramp_mode_str /= np.linalg.norm(ramp_mode_str)

    return ramp_mode_str, ramp_mode_alm
    

def episode(
    env,
    inp_dim,
    hid_dim,
    action_dim,
    actor, 
    check_path,
    frameskips,
    act_scale,
    act_bias,
    num_conditions,
    alm_hid_units,
    perturb,
    perturb_strength=None,
    silenced_region=None
):

    checkpoint = torch.load(check_path)
    _actor = actor(inp_dim, hid_dim, action_dim, act_scale, act_bias).cuda()
    _actor.load_state_dict(checkpoint['agent_state_dict'])

    str_activity = {}
    alm_activity = {}

    ### STEPS PER EPISODE ###
    for conditions in range(num_conditions):

        h_prev = torch.zeros(size=(1, 1, hid_dim), device="cuda")
        state = env.reset(conditions) 
        episode_steps = 0
        str_activity[conditions] = []
        alm_activity[conditions] = []

        for t in range(int(env.max_timesteps / frameskips)):

            with torch.no_grad():
                action, h_current = select_action(_actor, state, h_prev, evaluate=True)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(frameskips):
                str_activity[conditions].append(h_prev.squeeze().cpu().numpy())
                alm_activity[conditions].append(np.array(state))
                next_state, _, done = env.step(episode_steps, action, conditions)
                episode_steps += 1
                if done:
                    break
            
            if perturb == True and t > 40 and t < 60 and silenced_region == "alm":
                next_state = list(perturb_strength * torch.ones(size=(len(next_state),)))
            if perturb == True and t > 10 and t < 40 and silenced_region == "str":
                h_current = perturb_strength * torch.ones_like(h_current)

            state = next_state
            h_prev = h_current

            if done:
                print(env.lick)
                break

    # reset tracking variables
    print(np.array(str_activity[0]).shape)
    print(np.array(alm_activity[0]).shape)
    A_agent = gaussian_filter1d(np.array(str_activity[0]), 2, axis=0)
    A_alm = gaussian_filter1d(np.array(alm_activity[0]), 2, axis=0)

    return A_agent, A_alm

if __name__ == "__main__":

    env = Lick_Env_Cont(ACTION_DIM, TIMESTEPS, THRESH, DT, BETA, BG_SCALE, TRAJECTORY, ALM_NET_PATH, ALM_HID_UNITS)
    perturbed_psth_str_1, perturbed_psth_alm_1 = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, True, 0.0, "alm")
    perturbed_psth_str_2, perturbed_psth_alm_2 = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, True, 0.035, "alm")
    perturbed_psth_str_3, perturbed_psth_alm_3 = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, True, 0.065, "alm")
    normal_psth_str, normal_psth_alm = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, False)

    ramp_mode_str, ramp_mode_alm = get_ramp_mode(normal_psth_str, normal_psth_alm)

    perturbed_psth_str_1 = perturbed_psth_str_1 @ ramp_mode_str
    perturbed_psth_str_2 = perturbed_psth_str_2 @ ramp_mode_str
    perturbed_psth_str_3 = perturbed_psth_str_3 @ ramp_mode_str
    normal_psth_str = normal_psth_str @ ramp_mode_str

    perturbed_psth_alm_1 = perturbed_psth_alm_1 @ ramp_mode_alm
    perturbed_psth_alm_2 = perturbed_psth_alm_2 @ ramp_mode_alm
    perturbed_psth_alm_3 = perturbed_psth_alm_3 @ ramp_mode_alm
    normal_psth_alm = normal_psth_alm @ ramp_mode_alm

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")
    x_p1 = np.linspace(0, 1.1, perturbed_psth_alm_1.shape[0])
    x_p2 = np.linspace(0, 1.1, perturbed_psth_alm_2.shape[0])
    x_p3 = np.linspace(0, 1.1, perturbed_psth_alm_3.shape[0])
    x_u = np.linspace(0, 1.1, normal_psth_alm.shape[0])
    plt.plot(x_u, normal_psth_alm, label="Unperturbed Network", linewidth=4, color="#0F45A0")
    plt.plot(x_p1, perturbed_psth_alm_1, label="Strong Silencing", linewidth=4, color="#317FFF")
    plt.plot(x_p2, perturbed_psth_alm_2, label="Medium Silencing", linewidth=4, color="#659FFF")
    plt.plot(x_p3, perturbed_psth_alm_3, label="Weak Silencing", linewidth=4, color="#97BEFF")
    plt.set_cmap("Blues") 
    plt.xlabel("Time")
    plt.ylabel("Ramp Mode Projection")
    plt.title("ALM Activity - ALM Silencing")
    plt.legend(fontsize="16")
    plt.show()

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")
    x_p1 = np.linspace(0, 1.1, perturbed_psth_str_1.shape[0])
    x_p2 = np.linspace(0, 1.1, perturbed_psth_str_2.shape[0])
    x_p3 = np.linspace(0, 1.1, perturbed_psth_str_3.shape[0])
    x_u = np.linspace(0, 1.1, normal_psth_str.shape[0])
    plt.plot(x_u, normal_psth_str, label="Unperturbed Network", linewidth=4, color="#0F45A0")
    plt.plot(x_p1, perturbed_psth_str_1, label="Strong Silencing", linewidth=4, color="#317FFF")
    plt.plot(x_p2, perturbed_psth_str_2, label="Medium Silencing", linewidth=4, color="#659FFF")
    plt.plot(x_p3, perturbed_psth_str_3, label="Weak Silencing", linewidth=4, color="#97BEFF")
    plt.set_cmap("Blues") 
    plt.xlabel("Time")
    plt.ylabel("Ramp Mode Projection")
    plt.title("STR Activity - ALM Silencing")
    plt.legend(fontsize="16")
    plt.show()

    perturbed_psth_str_1, perturbed_psth_alm_1 = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, True, 0.0, "str")
    perturbed_psth_str_2, perturbed_psth_alm_2 = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, True, 0.035, "str")
    perturbed_psth_str_3, perturbed_psth_alm_3 = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, True, 0.065, "str")
    normal_psth_str, normal_psth_alm = episode(env, INP_DIM, HID_DIM, ACTION_DIM, Actor, CHECK_PATH, FRAMESKIP, ACT_SCALE, ACT_BIAS, NUM_CONDITIONS, ALM_HID_UNITS, False)

    ramp_mode_str, ramp_mode_alm = get_ramp_mode(normal_psth_str, normal_psth_alm)

    perturbed_psth_str_1 = perturbed_psth_str_1 @ ramp_mode_str
    perturbed_psth_str_2 = perturbed_psth_str_2 @ ramp_mode_str
    perturbed_psth_str_3 = perturbed_psth_str_3 @ ramp_mode_str
    normal_psth_str = normal_psth_str @ ramp_mode_str

    perturbed_psth_alm_1 = perturbed_psth_alm_1 @ ramp_mode_alm
    perturbed_psth_alm_2 = perturbed_psth_alm_2 @ ramp_mode_alm
    perturbed_psth_alm_3 = perturbed_psth_alm_3 @ ramp_mode_alm
    normal_psth_alm = normal_psth_alm @ ramp_mode_alm

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")
    x_p1 = np.linspace(0, 1.1, perturbed_psth_alm_1.shape[0])
    x_p2 = np.linspace(0, 1.1, perturbed_psth_alm_2.shape[0])
    x_p3 = np.linspace(0, 1.1, perturbed_psth_alm_3.shape[0])
    x_u = np.linspace(0, 1.1, normal_psth_alm.shape[0])
    plt.plot(x_u, normal_psth_alm, label="Unperturbed Network", linewidth=4, color="#0F45A0")
    plt.plot(x_p1, perturbed_psth_alm_1, label="Strong Silencing", linewidth=4, color="#317FFF")
    plt.plot(x_p2, perturbed_psth_alm_2, label="Medium Silencing", linewidth=4, color="#659FFF")
    plt.plot(x_p3, perturbed_psth_alm_3, label="Weak Silencing", linewidth=4, color="#97BEFF")
    plt.set_cmap("Blues") 
    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.title("ALM Activity - STR Silencing")
    plt.legend(fontsize="16")
    plt.show()

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")
    x_p1 = np.linspace(0, 1.1, perturbed_psth_str_1.shape[0])
    x_p2 = np.linspace(0, 1.1, perturbed_psth_str_2.shape[0])
    x_p3 = np.linspace(0, 1.1, perturbed_psth_str_3.shape[0])
    x_u = np.linspace(0, 1.1, normal_psth_str.shape[0])
    plt.plot(x_u, normal_psth_str, label="Unperturbed Network", linewidth=4, color="#0F45A0")
    plt.plot(x_p1, perturbed_psth_str_1, label="Strong Silencing", linewidth=4, color="#317FFF")
    plt.plot(x_p2, perturbed_psth_str_2, label="Medium Silencing", linewidth=4, color="#659FFF")
    plt.plot(x_p3, perturbed_psth_str_3, label="Weak Silencing", linewidth=4, color="#97BEFF")
    plt.set_cmap("Blues") 
    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.title("STR Activity - STR Silencing")
    plt.legend(fontsize="16")
    plt.show()