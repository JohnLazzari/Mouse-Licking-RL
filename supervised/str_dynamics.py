import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_delay_data

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDITION = 0
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"

def get_perturbed_trajectories(rnn, len_seq, total_num_units, x_data, ITI_steps, start_silence, end_silence, str_mask, stim_strength):

    perturbed_acts = []
    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()
    xn = hn

    for t in range(len_seq[CONDITION] + 1000):

        if t < ITI_steps:
            inp = x_data[CONDITION:CONDITION+1, 0:1, :]
        else:
            inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]

        with torch.no_grad():        

            if t > start_silence and t < end_silence:
                inhib_stim = (stim_strength * str_mask).unsqueeze(0).unsqueeze(0)
            else:
                inhib_stim = torch.zeros(size=(1, 1, hn.shape[-1]), device="cuda")
                    
            _, hn, _, xn, _ = rnn(inp, hn, xn, inhib_stim, noise=False)
            
            perturbed_acts.append(hn)
    
    perturbed_acts = torch.concatenate(perturbed_acts, dim=1).clone().detach().cuda()

    return perturbed_acts

def main():
    
    ITI_steps = 1000
    extra_steps = 0
    start_silence = 600 + ITI_steps
    end_silence = 1100 + ITI_steps
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)

    str_mask = rnn.str_d1_mask

    total_num_units = HID_DIM * 6
    str_start = 0
    stn_start = HID_DIM*2
    snr_start = HID_DIM*3

    # Get input and output data
    x_data, y_data, len_seq = gather_delay_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    y_data = y_data.cuda()

    # Sample many hidden states to get pcs for dimensionality reduction
    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()
    xn = hn

    inhib_stim = torch.zeros(size=(1, x_data.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, _, act, _, _ = rnn(x_data[CONDITION:CONDITION+1, :, :], hn, xn, inhib_stim, noise=False)
    
    sampled_acts = act[0, 500:len_seq[CONDITION], :]

    d1_activity = sampled_acts[:, :int(HID_DIM/2)]
    d2_activity = sampled_acts[:, int(HID_DIM/2):HID_DIM]
    
    sampled_acts = sampled_acts.detach().cpu().numpy()
    d1_activity = d1_activity.detach().cpu().numpy()
    d2_activity = d2_activity.detach().cpu().numpy()

    d1_activity_peak = np.mean(d1_activity[-400:, :], axis=0)
    d2_activity_peak = np.mean(d2_activity[-400:, :], axis=0)

    disc = d1_activity_peak - d2_activity_peak
    disc = disc / np.linalg.norm(disc)

    # reduce d1 and d2 snr activity using PCA
    d2_snr_activity_reduced = d2_activity @ disc
    d1_snr_activity_reduced = d1_activity @ disc

    # Get perturbed trajectories
    d1_perturbed_trajectories = []
    d2_perturbed_trajectories = []
    stim_strengths = [0.25, 0.75, 1.25]

    for stim in stim_strengths:
        perturbed_act = get_perturbed_trajectories(rnn, len_seq, total_num_units, x_data, ITI_steps, start_silence, end_silence, str_mask, stim)
        d1_perturbed_trajectories.append(perturbed_act[0, 500:, :int(HID_DIM/2)])
        d2_perturbed_trajectories.append(perturbed_act[0, 500:, int(HID_DIM/2):HID_DIM])

    projected_perturbed_d1 = []
    projected_perturbed_d2 = []

    for d1, d2 in zip(d1_perturbed_trajectories, d2_perturbed_trajectories):

        projected_perturbed_d1.append(d1.detach().cpu().numpy() @ disc)
        projected_perturbed_d2.append(d2.detach().cpu().numpy() @ disc)
    
    plt.plot(d1_snr_activity_reduced, linewidth=4, color="red") 
    plt.plot(d2_snr_activity_reduced, linewidth=4, color="blue") 
    plt.show()
    
    for i in range(3):
        plt.plot(projected_perturbed_d1[i], linewidth=4, color=(1-i*0.25, 0.1, 0.1)) 
        plt.plot(projected_perturbed_d2[i], linewidth=4, color=(0.1, 0.1, 1-i*0.25)) 
    
    plt.show()

if __name__ == "__main__":
    main()