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
from utils import gather_delay_data, get_acts, get_ramp_mode, project_ramp_mode
import tqdm

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 26}
plt.rcParams['figure.figsize'] = [10, 8]
plt.rc('font', **font)

CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1.pth"
HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDS = 3
MODEL_TYPE = "d1" # d1d2, d1, stralm

def plot_silencing(len_seq, conds, rnn, hid_dim, x_data, title, silenced_region, evaluated_region, dt, use_label=False):

    ITI_steps = 1000

    if MODEL_TYPE == "d1d2" and evaluated_region == "alm":
        start = hid_dim*5
        end = hid_dim*6
    elif MODEL_TYPE == "d1d2" and evaluated_region == "str":
        # Only silence D1
        start = 0
        end = int(hid_dim/2)
    elif MODEL_TYPE == "stralm" and evaluated_region == "alm":
        start = hid_dim
        end = hid_dim*2
    elif MODEL_TYPE == "stralm" and evaluated_region == "str":
        start = 0
        end = hid_dim
    elif MODEL_TYPE == "d1" and evaluated_region == "alm":
        start = hid_dim*2
        end = hid_dim*3
    elif MODEL_TYPE == "d1" and evaluated_region == "str":
        start = 0
        end = hid_dim
    
    ramp_orig = {}
    ramp_silenced = {}

    for cond in range(conds):

        # activity without silencing
        acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, False, MODEL_TYPE)

        baseline_orig_control = np.mean(acts[500:1000, start:end], axis=0)
        peak_orig_control = np.mean(acts[1100 + 500*cond - 400 + ITI_steps:1100 + 500*cond + ITI_steps, start:end], axis=0)

        ramp_mode = get_ramp_mode(baseline_orig_control, peak_orig_control)
        projected_orig = project_ramp_mode(acts[:, start:end], ramp_mode)

        ramp_orig[cond] = projected_orig

        # activity with silencing
        acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, True, MODEL_TYPE, region=silenced_region)
        projected_silenced = project_ramp_mode(acts[:, start:end], ramp_mode)
        ramp_silenced[cond] = projected_silenced

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    xs_p = {}
    xs_u = {}
    for cond in range(conds):

        xs_p[cond] = np.linspace(-0.5, 1.1 + 0.5 * cond + (1000 * dt), ramp_silenced[cond].shape[0] - 500)
        xs_u[cond] = np.linspace(-0.5, 1.1 + 0.5 * cond + (700 * dt), ramp_orig[cond].shape[0] - 500)

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
    plt.show()

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    for cond in range(conds):
        plt.plot(xs_p[cond], ramp_silenced[cond][500:], linewidth=8, color=(1-cond*0.25, 0.1, 0.1))

    plt.xticks([])
    plt.tick_params(left=False, bottom=False) 
    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM).cuda()

    rnn.load_state_dict(checkpoint)

    x_data, _, len_seq = gather_delay_data(dt=0.001, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "ALM PSTH", "alm", "alm", DT, use_label=True)
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "ALM PSTH", "str", "alm", DT)
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "STR PSTH", "alm", "str", DT)
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "STR PSTH", "str", "str", DT)
    
if __name__ == "__main__":
    main()