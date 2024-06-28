import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM, RNN_MultiRegional_D1
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_inp_data, get_acts_control, get_acts_manipulation, get_ramp_mode, project_ramp_mode
import tqdm
import time

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 26}
plt.rcParams['figure.figsize'] = [10, 8]
plt.rc('font', **font)

HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDS = 3
MODEL_TYPE = "d1" # d1d2, d1, stralm
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}.pth"
SAVE_NAME_PATH = f"results/multi_regional_perturbations/{MODEL_TYPE}/"
CONSTRAINED = True
ITI_STEPS = 1000
START_SILENCE = 1600 # timepoint from start of trial to silence at
END_SILENCE = 2100 # timepoint from start of trial to end silencing

def plot_silencing(len_seq, 
                   conds, 
                   rnn, 
                   hid_dim, 
                   x_data, 
                   save_name_control, 
                   save_name_silencing, 
                   silenced_region, 
                   evaluated_region, 
                   dt, 
                   stim_strength, 
                   extra_steps_control,
                   extra_steps_silence,
                   use_label=False, 
                   ):


    if MODEL_TYPE == "d1d2" and evaluated_region == "alm":
        start = hid_dim*5
        end = hid_dim*6
    elif MODEL_TYPE == "d1d2" and evaluated_region == "str":
        start = 0
        end = int(hid_dim/2)
    elif MODEL_TYPE == "stralm" and evaluated_region == "alm":
        start = hid_dim
        end = hid_dim*2
    elif MODEL_TYPE == "stralm" and evaluated_region == "str":
        start = 0
        end = hid_dim
    elif MODEL_TYPE == "d1" and evaluated_region == "alm":
        start = hid_dim*3
        end = hid_dim*4
    elif MODEL_TYPE == "d1" and evaluated_region == "str":
        start = 0
        end = hid_dim
    
    ramp_orig = {}
    ramp_silenced = {}

    for cond in range(conds):

        # activity without silencing
        acts = get_acts_control(len_seq, 
                                rnn, 
                                hid_dim, 
                                x_data, 
                                cond, 
                                MODEL_TYPE, 
                                ITI_STEPS, 
                                extra_steps_control)

        baseline_orig_control = np.mean(acts[500:1000, start:end], axis=0)
        peak_orig_control = np.mean(acts[1100 + 500*cond - 400 + ITI_STEPS:1100 + 500*cond + ITI_STEPS, start:end], axis=0)

        ramp_mode = get_ramp_mode(baseline_orig_control, peak_orig_control)
        projected_orig = project_ramp_mode(acts[:, start:end], ramp_mode)

        ramp_orig[cond] = projected_orig

        # activity with silencing
        acts = get_acts_manipulation(len_seq, 
                                    rnn, 
                                    hid_dim, 
                                    x_data, 
                                    cond, 
                                    MODEL_TYPE, 
                                    ITI_STEPS,
                                    START_SILENCE,
                                    END_SILENCE,
                                    stim_strength, 
                                    extra_steps_silence, 
                                    silenced_region, 
                                    )

        projected_silenced = project_ramp_mode(acts[:, start:end], ramp_mode)
        ramp_silenced[cond] = projected_silenced

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    xs_p = {}
    xs_u = {}
    for cond in range(conds):

        xs_p[cond] = np.linspace(-0.5, 1.1 + 0.5 * cond + (extra_steps_silence * dt), ramp_silenced[cond].shape[0] - 500)
        xs_u[cond] = np.linspace(-0.5, 1.1 + 0.5 * cond + (extra_steps_control * dt), ramp_orig[cond].shape[0] - 500)

    for cond in range(conds):
        if use_label:
            plt.plot(xs_u[cond], ramp_orig[cond][500:], label=f"Lick Time {1.1 + 0.5 * cond}s", linewidth=8, color=(1-cond*0.25, 0.1, 0.1))
            plt.axvline(x=1.1 + 0.5 * cond, linestyle='--', color=(1-cond*0.25, 0.1, 0.1))
        else:
            plt.plot(xs_u[cond], ramp_orig[cond][500:], linewidth=8, color=(1-cond*0.25, 0.1, 0.1))

    if use_label:
        plt.xlabel("Time (s)")
        plt.ylabel("Ramp Mode Projection")
        plt.legend(loc="lower right")
    else:
        plt.xticks([])
        
    plt.tick_params(left=False, bottom=False) 
    plt.savefig(f"{save_name_control}.png")
    plt.close()

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    for cond in range(conds):
        plt.plot(xs_p[cond], ramp_silenced[cond][500:], linewidth=8, color=(1-cond*0.25, 0.1, 0.1))

    plt.xticks([])
    plt.tick_params(left=False, bottom=False) 
    plt.savefig(f"{save_name_silencing}.png")
    plt.close()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()

    rnn.load_state_dict(checkpoint)

    x_data, len_seq = gather_inp_data(dt=0.001, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    
    plot_silencing(len_seq, 
                   CONDS, 
                   rnn, 
                   HID_DIM, 
                   x_data, 
                   SAVE_NAME_PATH + "alm_activity_control", 
                   SAVE_NAME_PATH + "alm_activity_alm_silencing",
                   silenced_region="alm", 
                   evaluated_region="alm", 
                   dt=DT, 
                   stim_strength=-10, 
                   extra_steps_control=0,
                   extra_steps_silence=1000,
                   use_label=True)

    plot_silencing(len_seq, 
                   CONDS, 
                   rnn, 
                   HID_DIM, 
                   x_data, 
                   SAVE_NAME_PATH + "alm_activity_control", 
                   SAVE_NAME_PATH + "alm_activity_str_silencing",
                   silenced_region="str", 
                   evaluated_region="alm", 
                   dt=DT, 
                   stim_strength=-0.35, 
                   extra_steps_control=0,
                   extra_steps_silence=1000,
                   )

    plot_silencing(len_seq, 
                   CONDS, 
                   rnn, 
                   HID_DIM, 
                   x_data, 
                   SAVE_NAME_PATH + "str_activity_control", 
                   SAVE_NAME_PATH + "str_activity_alm_silencing",
                   silenced_region="alm", 
                   evaluated_region="str", 
                   dt=DT, 
                   stim_strength=-10,
                   extra_steps_control=0,
                   extra_steps_silence=1000,
                   )

    plot_silencing(len_seq, 
                   CONDS, 
                   rnn, 
                   HID_DIM, 
                   x_data, 
                   SAVE_NAME_PATH + "str_activity_control", 
                   SAVE_NAME_PATH + "str_activity_str_silencing",
                   silenced_region="str", 
                   evaluated_region="str", 
                   dt=DT, 
                   stim_strength=-0.35, 
                   extra_steps_control=0,
                   extra_steps_silence=1000,
                   )
    
if __name__ == "__main__":
    main()