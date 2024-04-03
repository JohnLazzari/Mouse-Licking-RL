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

def main():

    inp_dim = 2
    hid_dim = 32
    out_dim = 1
    cond = 0

    check_path = "checkpoints/rnn_goal_data_full_delay.pth"
    checkpoint = torch.load(check_path)
    
    # Create RNN
    rnn = RNN(inp_dim, hid_dim, out_dim)
    rnn.load_state_dict(checkpoint)

    y_data, x_data, len_seq = gather_delay_data()
    
    hn = torch.zeros(size=(1, 1, hid_dim))

    # ORIG
    acts = []
    for t in range(len_seq[cond]):
        with torch.no_grad():        
            out, hn, act = rnn(x_data[cond:cond+1, t:t+1, :], hn)
            print(out)
            acts.append(hn.squeeze().numpy())
    
    acts = np.array(acts)
    ramp = np.mean(acts[200:209, :], axis=0)
    baseline = np.mean(acts[30:40, :], axis=0)
    d_ramp = ramp - baseline
    projected_orig = acts @ d_ramp 

    # TEST
    hn = torch.zeros(size=(1, 1, hid_dim))
    inhibit_strength = 0.1
    acts = []
    for t in range(len_seq[cond]):
        with torch.no_grad():        
            out, hn, act = rnn(x_data[cond:cond+1, t:t+1, :], hn)
            if t > 100 and t < 140:
                hn = hn * inhibit_strength
            acts.append(hn.squeeze().numpy())
    
    acts = np.array(acts)
    ramp = np.mean(acts[200:209, :], axis=0)
    baseline = np.mean(acts[30:40, :], axis=0)
    d_ramp = ramp - baseline
    projected_perturbed = acts @ d_ramp 
    
    plt.plot(projected_orig[5:])
    plt.plot(projected_perturbed[5:])
    plt.show()
    

if __name__ == "__main__":
    main()