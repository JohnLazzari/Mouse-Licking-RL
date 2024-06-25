import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
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
LR = 1e-3
CONDITION = 0
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"

class Vec(nn.Module):
    def __init__(self):
        super(Vec, self).__init__()

        self.subspace = nn.Parameter(torch.empty(size=(HID_DIM,)))
        nn.init.uniform_(self.subspace, -0.1, 0.1)
    
    def forward(self, W_str, W_alm, thal_activity):
        
        str_projection = W_str @ self.subspace
        alm_projection = W_alm @ self.subspace
        thal_projection = thal_activity @ self.subspace

        return str_projection, alm_projection, thal_projection

def get_perturbed_trajectories(rnn, len_seq, total_num_units, x_data, ITI_steps, start_silence, end_silence, stim):

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
                inhib_stim = stim.unsqueeze(0).unsqueeze(0)
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
    
    thal2alm = F.relu(rnn.thal2alm_weight_l0_hh).detach()
    thal2str = rnn.thal2str_mask * F.relu(rnn.thal2str_weight_l0_hh).detach()
    
    sampled_acts = act[0, 500:len_seq[CONDITION], :].detach().clone()
    thal_activity = sampled_acts[:, HID_DIM*4:HID_DIM*5].detach().clone().cuda()
    
    # Initiate vectors
    thal2alm_vec = Vec().cuda()
    thal2str_vec = Vec().cuda()

    # Train first vector, excite ALM and no activity for STR
    thal2alm_optim = optim.Adam(thal2alm_vec.parameters(), lr=LR)
    thal2str_optim = optim.Adam(thal2str_vec.parameters(), lr=LR)

    losses_thal2alm = []
    losses_thal2str = []

    for iter in range(100000):
        
        str_act, alm_act, thal_projection = thal2alm_vec(thal2str, thal2alm, thal_activity)
        loss = (torch.sum(str_act)**2) + (torch.sum(alm_act) - 1)**2 + (torch.sum(torch.sum(thal_projection)) - 1)**2
        losses_thal2alm.append(loss.item())
        print(f"Loss at iter {iter}: {loss}")
        thal2alm_optim.zero_grad()
        loss.backward()
        thal2alm_optim.step()

    for iter in range(100000):
        
        str_act, alm_act, thal_projection = thal2str_vec(thal2str, thal2alm, thal_activity)
        loss = (torch.sum(str_act) - 1)**2 + (torch.sum(alm_act)**2) + (torch.sum(torch.sum(thal_projection)) - 1)**2
        losses_thal2str.append(loss.item())
        print(f"Loss at iter {iter}: {loss}")
        thal2str_optim.zero_grad()
        loss.backward()
        thal2str_optim.step()

    plt.plot(losses_thal2alm, color="green", linewidth=8)
    plt.xlabel("Iter")
    plt.ylabel("MSE")
    plt.show()

    plt.plot(losses_thal2str, color="red", linewidth=8)
    plt.show()

    thal2alm_subspace = (thal2alm_vec.subspace / (torch.linalg.norm(thal2alm_vec.subspace))).clone().detach()
    thal2str_subspace = (thal2str_vec.subspace / (torch.linalg.norm(thal2str_vec.subspace))).clone().detach()

    inner_product = torch.inner(thal2alm_subspace, thal2str_subspace)
    print(inner_product)

    # Check to see if thalamus ramps along these directions
    projected_thal2alm = thal_activity @ thal2alm_subspace
    projected_thal2str = thal_activity @ thal2str_subspace

    x = np.linspace(-0.5, (len_seq[0] - 1000) * DT, len_seq[0] - 500)
    plt.axvline(x=0.0, linestyle='--', color='black', label="Cue")
    plt.plot(x, projected_thal2alm.detach().cpu().numpy(), linewidth=8, color="green")
    plt.plot(x, projected_thal2str.detach().cpu().numpy(), linewidth=8, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Projected Activity")
    plt.show()

    # Get perturbed trajectories
    thal2alm_perturbed_trajectories = []
    thal2str_perturbed_trajectories = []

    # First get perturbed trajectory of thal2alm vector, should see integration happening while alm is generally suppressed
    print(-10 * thal2alm_subspace.cpu())
    stim = torch.cat([torch.zeros(size=(HID_DIM*4,)),
                        -10 * thal2alm_subspace.cpu(), 
                        torch.zeros(size=(HID_DIM,))]).cuda()

    perturbed_act = get_perturbed_trajectories(rnn, len_seq, total_num_units, x_data, ITI_steps, start_silence, end_silence, stim)
    str_perturbed_trajectories_thal2alm = perturbed_act[0, 500:, str_start:str_start+HID_DIM]
    alm_perturbed_trajectories_thal2alm = perturbed_act[0, 500:, HID_DIM*5:]

    x = np.linspace(-0.5, (len_seq[0]) * DT, len_seq[0] - 500 + 1000)
    plt.axvline(x=0.0, linestyle='--', color='black', label="Cue")
    plt.plot(x, np.mean(alm_perturbed_trajectories_thal2alm.detach().cpu().numpy(), axis=1), linewidth=8, color="green")
    plt.plot(x, np.mean(str_perturbed_trajectories_thal2alm.detach().cpu().numpy(), axis=1), linewidth=8, color="red")
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Activity")
    plt.show()

    projected_perturbed_d1 = []
    projected_perturbed_d2 = []

    for d1, d2 in zip(d1_perturbed_trajectories, d2_perturbed_trajectories):

        projected_perturbed_d1.append(d1.detach().cpu().numpy() @ disc)
        projected_perturbed_d2.append(d2.detach().cpu().numpy() @ disc)
    
    for i in range(3):
        plt.plot(projected_perturbed_d1[i], linewidth=4, color=(1-i*0.25, 0.1, 0.1)) 
        plt.plot(projected_perturbed_d2[i], linewidth=4, color=(0.1, 0.1, 1-i*0.25)) 
    
    plt.show()

if __name__ == "__main__":
    main()