import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def NormalizeData(data, min, max):
    return (data - min) / (max - min)

def gather_delay_data():
    
    lick_struct = {}
    ramp_inp = {}

    # Condition 1
    lick_struct[0] = torch.zeros(size=(210,)).unsqueeze(1)
    lick_struct[0][194:209] = 1

    # Condition 2
    lick_struct[1] = torch.zeros(size=(240,)).unsqueeze(1)
    lick_struct[1][224:239] = 1

    # Condition 3
    lick_struct[2] = torch.zeros(size=(270,)).unsqueeze(1)
    lick_struct[2][254:269] = 1

    for cond in range(3):
        ramp = torch.linspace(0, 1, int((1.1 + (.3*cond)) / 0.01), dtype=torch.float32).unsqueeze(1)
        baseline = torch.zeros(size=(100, 1))
        ramp_inp[cond] = torch.cat((baseline, ramp), dim=0)

    len_seq = [210, 240, 270]
    cue_inp = torch.zeros(size=(3, 270, 1))
    cue_inp[:, 99, :] = 1
    ramp_inp_total = pad_sequence([ramp_inp[0], ramp_inp[1], ramp_inp[2]], batch_first=True)
    total_inp = torch.cat((ramp_inp_total, cue_inp), dim=-1)
    lick_seq_total = pad_sequence([lick_struct[0], lick_struct[1], lick_struct[2]], batch_first=True)

    return lick_seq_total, total_inp, len_seq

def get_acts(len_seq, rnn, hid_dim, x_data, cond, perturbation, perturbation_strength=None):

    acts = []
    hn = torch.zeros(size=(1, 1, hid_dim))

    for t in range(len_seq):
        with torch.no_grad():        
            out, hn, act = rnn(x_data[cond:cond+1, t:t+1, :], hn)
            print(out)
            if perturbation == True and t > 110 and t < 140:
                hn = perturbation_strength * hn
            acts.append(hn.squeeze().numpy())
    
    return np.array(acts)

def main():

    inp_dim = 2
    hid_dim = 100
    out_dim = 1
    cond = 0

    check_path = "checkpoints/rnn_goal_data_100n_delay.pth"
    checkpoint = torch.load(check_path)
    
    # Create RNN
    rnn = RNN(inp_dim, hid_dim, out_dim)
    rnn.load_state_dict(checkpoint)

    y_data, x_data, len_seq = gather_delay_data()
    
    # ORIG
    acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, False)
    ramp_psth_orig = np.mean(acts[100:209, :], axis=1)

    # TEST
    acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, True, perturbation_strength=0.0)
    ramp_psth_silenced_1 = np.mean(acts[100:209, :], axis=1)

    acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, True, perturbation_strength=0.25)
    ramp_psth_silenced_2 = np.mean(acts[100:209, :], axis=1)

    acts = get_acts(len_seq[cond], rnn, hid_dim, x_data, cond, True, perturbation_strength=0.5)
    ramp_psth_silenced_3 = np.mean(acts[100:209, :], axis=1)
    
    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")
    x_p1 = np.linspace(0, 1.1, ramp_psth_silenced_1.shape[0])
    x_p2 = np.linspace(0, 1.1, ramp_psth_silenced_2.shape[0])
    x_p3 = np.linspace(0, 1.1, ramp_psth_silenced_3.shape[0])
    x_u = np.linspace(0, 1.1, ramp_psth_orig.shape[0])
    plt.plot(x_u, ramp_psth_orig, label="Unperturbed Network", linewidth=4, color="#0F45A0")
    plt.plot(x_p1, ramp_psth_silenced_1, label="Strong Silencing", linewidth=4, color="#317FFF")
    plt.plot(x_p2, ramp_psth_silenced_2, label="Medium Silencing", linewidth=4, color="#659FFF")
    plt.plot(x_p3, ramp_psth_silenced_3, label="Weak Silencing", linewidth=4, color="#97BEFF")
    plt.set_cmap("Blues") 
    plt.xlabel("Time")
    plt.ylabel("Firing Rate")
    plt.title("ALM Activity Without Feedback")
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()