import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_inp_data, get_acts_manipulation

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 16}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 4 # set the value globally
plt.rcParams['figure.figsize'] = [10, 8]

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
DT = 1e-3
CONDITION = 0
MODEL_TYPE = "d1d2"
CONSTRAINED = True
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}.pth"

def main():
    
    checkpoint = torch.load(CHECK_PATH)
    
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
        rnn.load_state_dict(checkpoint)
        total_num_units = HID_DIM * 6
        str_start = int(HID_DIM/4)
        str_end = int(HID_DIM/2)
        inp_mask = rnn.strthal_mask
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
        rnn.load_state_dict(checkpoint)
        total_num_units = HID_DIM * 4
        str_start = int(HID_DIM/2)
        str_end = HID_DIM
        inp_mask = rnn.strthal_mask
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
        rnn.load_state_dict(checkpoint)
        total_num_units = HID_DIM * 2
        str_start = 0
        str_end = HID_DIM
        inp_mask = rnn.str_d1_mask

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()

    # Inhibitory/excitatory stimulus to network, designed as an input current
    # Does this for a single condition, len_seq should be a single number for the chosen condition, and x_data should be [1, len_seq, :]
    inhib_stim_pre = 0 * torch.ones(size=(3, 1000, total_num_units), device="cuda") * rnn.alm_mask
    inhib_stim_silence = -10 * torch.ones(size=(3, 3100 - 1000, total_num_units), device="cuda") * rnn.alm_mask
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence], dim=1)

    # Get original trajectory
    with torch.no_grad():
        _, act = rnn(x_data, hn, inhib_stim, noise=False)
    
    vel_const = rnn.t_const * (x_data @ F.hardtanh(rnn.inp_weight, 1e-15, 1) * inp_mask)
    vel_const = np.mean(vel_const[:, 1000:, str_start:str_end].detach().cpu().numpy(), axis=-1)

    int_activity = act[:, 999:, str_start:str_end]
    int_activity_d = (int_activity[:, 1:, :] - int_activity[:, :-1, :])
    int_activity_d = np.mean(int_activity_d.detach().cpu().numpy(), axis=-1)

    abs_error = np.abs(int_activity_d - vel_const) 

    plt.axvline(x=0.0, linestyle='--', color='black', label="Cue")
    plt.plot(abs_error[0, :len_seq[0] - 1000], linewidth=10, color=(1-0*0.25, 0.1, 0.1)) 
    plt.plot(abs_error[1, :len_seq[1] - 1000], linewidth=10, color=(1-1*0.25, 0.1, 0.1)) 
    plt.plot(abs_error[2, :len_seq[2] - 1000], linewidth=10, color=(1-2*0.25, 0.1, 0.1)) 
    plt.xticks([])
    plt.show()

if __name__ == "__main__":
    main()