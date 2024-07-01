import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_inp_data

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 20}
plt.rc('font', **font)

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
DT = 1e-3
CONDITION = 0
MODEL_TYPE = "d1"
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}.pth"

def main():
    
    checkpoint = torch.load(CHECK_PATH)
    
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
        rnn.load_state_dict(checkpoint)
        total_num_units = HID_DIM * 6
        str_start = 0
        inp_mask = rnn.strthal_mask
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM).cuda()
        rnn.load_state_dict(checkpoint)
        total_num_units = HID_DIM * 4
        str_start = 0
        inp_mask = rnn.strthal_mask
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM).cuda()
        rnn.load_state_dict(checkpoint)
        total_num_units = HID_DIM * 2
        str_start = 0
        inp_mask = rnn.str_d1_mask

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    # Sample many hidden states to get pcs for dimensionality reduction
    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()
    xn = hn

    inhib_stim = torch.zeros(size=(1, x_data.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, act, _, _ = rnn(x_data, hn, xn, inhib_stim, noise=False)
    
    vel_const = rnn.t_const * (x_data @ F.hardtanh(rnn.inp_weight, 1e-15, 1) * inp_mask)
    vel_const = vel_const[:, 1000:, str_start:HID_DIM].detach().cpu().numpy()

    int_activity = act[:, 1000:, str_start:HID_DIM]
    int_activity_d = int_activity[:, 1:, :] - int_activity[:, :-1, :]
    int_activity_d = int_activity_d.detach().cpu().numpy()

    x_0 = np.linspace(0, (len_seq[0] - 1000) * DT, len_seq[0] - 1000)
    x_1 = np.linspace(0, (len_seq[1] - 1000) * DT, len_seq[1] - 1000)
    x_2 = np.linspace(0, (len_seq[2] - 1000) * DT, len_seq[2] - 1000)

    plt.axvline(x=0.0, linestyle='--', color='black', label="Cue")
    plt.plot(vel_const[0, :len_seq[0] - 1000], '--', linewidth=6, color=(1-0*0.25, 0.1, 0.1)) 
    plt.plot(vel_const[1, :len_seq[1] - 1000], '--', linewidth=6, color=(1-1*0.25, 0.1, 0.1)) 
    plt.plot(vel_const[2, :len_seq[2] - 1000], '--', linewidth=6, color=(1-2*0.25, 0.1, 0.1)) 
    plt.plot(int_activity_d[0, :len_seq[0] - 1001], linewidth=6, color=(1-0*0.25, 0.1, 0.1)) 
    plt.plot(int_activity_d[1, :len_seq[1] - 1001], linewidth=6, color=(1-1*0.25, 0.1, 0.1)) 
    plt.plot(int_activity_d[2, :len_seq[2] - 1001], linewidth=6, color=(1-2*0.25, 0.1, 0.1)) 
    plt.xlabel("Time (s)")
    plt.ylabel("Mean Activity")
    plt.show()

if __name__ == "__main__":
    main()