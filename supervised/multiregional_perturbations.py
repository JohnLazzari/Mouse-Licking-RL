import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_delay_data, get_acts, get_ramp

CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_gating.pth"
INP_DIM = 1
HID_DIM = 512
OUT_DIM = 1
DT = 1e-3
CONDS = 3

def plot_silencing(len_seq, conds, rnn, hid_dim, x_data, title, silenced_region, evaluated_region, dt):

    if evaluated_region == "alm":
        start = int(hid_dim*(3/4))
        end = hid_dim
    elif evaluated_region == "str":
        start = 0
        end = int(hid_dim/4)
    
    ramp_psth_orig = {}
    ramp_psth_silenced = {}

    for cond in range(conds):
    
        # Original PSTH without silencing
        acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, False)
        ramp_psth_orig[cond] = np.mean(acts[:, start:end], axis=1)

        # PSTH with silencing
        acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, True, perturbation_strength=0.0, region=silenced_region)
        ramp_psth_silenced[cond] = np.mean(acts[:, start:end], axis=1)
    

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    xs_p = {}
    xs_u = {}
    for cond in range(conds):

        xs_p[cond] = np.linspace(0, int((1.1 + 0.5 * cond) / dt) + 400, ramp_psth_silenced[cond].shape[0])
        xs_u[cond] = np.linspace(0, int((1.1 + 0.5 * cond) / dt), ramp_psth_orig[cond].shape[0])

    for cond in range(conds):
        plt.plot(xs_u[cond], ramp_psth_orig[cond], label=f"Unperturbed Network Cond {cond}", linewidth=4)

    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.ylim([0, 1])
    plt.title(title)
    plt.legend()
    plt.show()

    for cond in range(conds):
        plt.plot(xs_p[cond], ramp_psth_silenced[cond], label=f"Silencing Cond {cond}", linewidth=4)

    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.ylim([0, 1])
    plt.title(title)
    plt.legend()
    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)

    x_data, _, len_seq = gather_delay_data(dt=0.001, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "ALM PSTH", "alm", "alm", DT)
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "ALM PSTH", "str", "alm", DT)
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "STR PSTH", "alm", "str", DT)
    plot_silencing(len_seq, CONDS, rnn, HID_DIM, x_data, "STR PSTH", "str", "str", DT)
    
if __name__ == "__main__":
    main()