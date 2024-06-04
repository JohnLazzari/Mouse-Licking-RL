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

CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit.pth"
INP_DIM = 1
HID_DIM = 512
OUT_DIM = 1
DT = 1e-3

def plot_silencing(len_seq, rnn, hid_dim, x_data, title, silenced_region, evaluated_region):

    if evaluated_region == "alm":
        start = int(hid_dim*(3/4))
        end = hid_dim
    elif evaluated_region == "str":
        start = 0
        end = int(hid_dim/4)
    
    # Original PSTH without silencing
    acts = get_acts(len_seq[0], rnn, hid_dim, x_data, 0, False)
    ramp_psth_orig_1 = np.mean(acts[:, start:end], axis=1)

    acts = get_acts(len_seq[1], rnn, hid_dim, x_data, 1, False)
    ramp_psth_orig_2 = np.mean(acts[:, start:end], axis=1)

    acts = get_acts(len_seq[2], rnn, hid_dim, x_data, 2, False)
    ramp_psth_orig_3 = np.mean(acts[:, start:end], axis=1)

    # PSTH with silencing
    acts = get_acts(len_seq[0], rnn, hid_dim, x_data, 0, True, perturbation_strength=0.0, region=silenced_region)
    ramp_psth_silenced_1 = np.mean(acts[:, start:end], axis=1)

    acts = get_acts(len_seq[1], rnn, hid_dim, x_data, 1, True, perturbation_strength=0.0, region=silenced_region)
    ramp_psth_silenced_2 = np.mean(acts[:, start:end], axis=1)

    acts = get_acts(len_seq[2], rnn, hid_dim, x_data, 2, True, perturbation_strength=0.0, region=silenced_region)
    ramp_psth_silenced_3 = np.mean(acts[:, start:end], axis=1)

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    x_p1 = np.linspace(0, 1.5, ramp_psth_silenced_1.shape[0])
    x_p2 = np.linspace(0, 2.0, ramp_psth_silenced_2.shape[0])
    x_p3 = np.linspace(0, 2.5, ramp_psth_silenced_3.shape[0])

    x_u1 = np.linspace(0, 1.1, ramp_psth_orig_1.shape[0])
    x_u2 = np.linspace(0, 1.6, ramp_psth_orig_2.shape[0])
    x_u3 = np.linspace(0, 2.1, ramp_psth_orig_3.shape[0])

    plt.plot(x_u1, ramp_psth_orig_1, label="Unperturbed Network Cond 1", linewidth=4, color="#0F45A0")
    plt.plot(x_u2, ramp_psth_orig_2, label="Unperturbed Network Cond 2", linewidth=4, color="#0F45A0")
    plt.plot(x_u3, ramp_psth_orig_3, label="Unperturbed Network Cond 3", linewidth=4, color="#0F45A0")

    plt.set_cmap("Blues") 
    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.ylim([0, 1])
    plt.title(title)
    plt.legend()
    plt.show()

    plt.plot(x_p1, ramp_psth_silenced_1, label="Silencing Cond 1", linewidth=4, color="#317FFF")
    plt.plot(x_p2, ramp_psth_silenced_2, label="Silencing Cond 2", linewidth=4, color="#659FFF")
    plt.plot(x_p3, ramp_psth_silenced_3, label="Silencing Cond 3", linewidth=4, color="#97BEFF")

    plt.set_cmap("Blues") 
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
    
    plot_silencing(len_seq, rnn, HID_DIM, x_data, "ALM PSTH", "alm", "alm")
    plot_silencing(len_seq, rnn, HID_DIM, x_data, "ALM PSTH", "str", "alm")
    plot_silencing(len_seq, rnn, HID_DIM, x_data, "STR PSTH", "alm", "str")
    plot_silencing(len_seq, rnn, HID_DIM, x_data, "STR PSTH", "str", "str")
    
if __name__ == "__main__":
    main()