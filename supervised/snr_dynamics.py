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
font = {'size' : 20}
plt.rc('font', **font)

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

    for t in range(len_seq[CONDITION]+1000):

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

    str_mask = rnn.str_d2_mask

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
        _, _, act, _, _ = rnn(x_data, hn, xn, inhib_stim, noise=False)
    
    sampled_acts = act[:, 500:, :]
    snr_act = sampled_acts[:, :, HID_DIM*2:HID_DIM*3]
    snr_act = np.mean(snr_act.detach().cpu().numpy(), axis=2)
    print(snr_act.shape)

    plt.plot(snr_act[0])
    plt.plot(snr_act[1])
    plt.plot(snr_act[2])
    plt.show()

    stn2snr = F.relu(rnn.stn2snr_weight_l0_hh)
    str2snr = (rnn.str2snr_mask * F.relu(rnn.str2snr_weight_l0_hh)) @ rnn.str2snr_D

    act_conds_d1 = []
    act_conds_d2 = []
    for i in range(3):
        d2_snr_activity = torch.matmul(stn2snr, sampled_acts[i, :, stn_start:stn_start+HID_DIM].T).T
        act_conds_d2.append(d2_snr_activity.detach().cpu().numpy())
        d1_snr_activity = torch.matmul(str2snr, sampled_acts[i, :, str_start:str_start+HID_DIM].T).T
        act_conds_d1.append(d1_snr_activity.detach().cpu().numpy())
    
    act_conds_d1 = np.array(act_conds_d1)
    act_conds_d2 = np.array(act_conds_d2)
    
    sampled_acts = sampled_acts.detach().cpu().numpy()
    d2_snr_activity = d2_snr_activity.detach().cpu().numpy()
    d1_snr_activity = d1_snr_activity.detach().cpu().numpy()

    # reduce d1 and d2 snr activity using PCA
    d2_snr_activity_reduced = np.mean(act_conds_d2, axis=2)
    d1_snr_activity_reduced = np.mean(act_conds_d1, axis=2)

    x_0 = np.linspace(-0.5, (len_seq[0] - 1000) * DT, len_seq[0] - 500)
    x_1 = np.linspace(-0.5, (len_seq[1] - 1000) * DT, len_seq[1] - 500)
    x_2 = np.linspace(-0.5, (len_seq[2] - 1000) * DT, len_seq[2] - 500)

    plt.axvline(x=0.0, linestyle='--', color='black', label="Cue")
    plt.plot(x_0, d1_snr_activity_reduced[0, :len_seq[0] - 500], linewidth=6, color=(1-0*0.25, 0.1, 0.1)) 
    plt.plot(x_1, d1_snr_activity_reduced[1, :len_seq[1] - 500], linewidth=6, color=(1-1*0.25, 0.1, 0.1)) 
    plt.plot(x_2, d1_snr_activity_reduced[2, :len_seq[2] - 500], linewidth=6, color=(1-2*0.25, 0.1, 0.1)) 
    plt.plot(x_0, d2_snr_activity_reduced[0, :len_seq[0] - 500], linewidth=6, color=(0.1, 0.1, 1-0*0.25)) 
    plt.plot(x_1, d2_snr_activity_reduced[1, :len_seq[1] - 500], linewidth=6, color=(0.1, 0.1, 1-1*0.25)) 
    plt.plot(x_2, d2_snr_activity_reduced[2, :len_seq[2] - 500], linewidth=6, color=(0.1, 0.1, 1-2*0.25)) 
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Activity")
    plt.show()

    # Get perturbed trajectories
    d1_perturbed_trajectories = []
    d2_perturbed_trajectories = []
    stim_strengths = [0.5, 1.5, 2.5]

    for stim in stim_strengths:
        perturbed_act = get_perturbed_trajectories(rnn, len_seq, total_num_units, x_data, ITI_steps, start_silence, end_silence, str_mask, stim)
        d1_perturbed_trajectories.append(perturbed_act[0, 500:len_seq[CONDITION], str_start:str_start+HID_DIM])
        d2_perturbed_trajectories.append(perturbed_act[0, 500:len_seq[CONDITION], stn_start:stn_start+HID_DIM])

    projected_perturbed_d1 = []
    projected_perturbed_d2 = []

    for d1, d2 in zip(d1_perturbed_trajectories, d2_perturbed_trajectories):

        d1_snr_activity_perturbed = (str2snr @ d1.T).T
        projected_perturbed_d1.append(np.mean(d1_snr_activity_perturbed.detach().cpu().numpy(), axis=1))

        d2_snr_activity_perturbed = (stn2snr @ d2.T).T
        projected_perturbed_d2.append(np.mean(d2_snr_activity_perturbed.detach().cpu().numpy(), axis=1))
    
    for i in range(3):
        plt.plot(projected_perturbed_d1[i], linewidth=4, color=(1-i*0.25, 0.1, 0.1)) 
        plt.plot(projected_perturbed_d2[i], linewidth=4, color=(0.1, 0.1, 1-i*0.25)) 
    
    plt.show()

if __name__ == "__main__":
    main()