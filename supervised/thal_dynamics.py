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
from utils import gather_inp_data, get_acts_manipulation

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 20}
plt.rc('font', **font)

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
DT = 1e-3
CONDITION = 0
START_SILENCE = 1600
END_SILENCE = 2200
MODEL_TYPE = "d1d2"
STIM_STRENGTH = -10
REGION_TO_SILENCE = "alm"
EXTRA_STEPS = 1000
ITI_STEPS = 1000
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2_alm2thal.pth"

def main():
    
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)

    total_num_units = HID_DIM * 6 + INP_DIM
    str_start = 0
    stn_start = HID_DIM*2
    snr_start = HID_DIM*3

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    iti_inp, cue_inp = x_data
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Sample many hidden states to get pcs for dimensionality reduction
    hn = torch.zeros(size=(1, 5, total_num_units)).cuda()

    inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, act = rnn(iti_inp, cue_inp, hn, inhib_stim, noise=False)
    
    thal2str = F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-10, 1).detach().cpu().numpy()
    thal_activity = act[:, :, HID_DIM*4:HID_DIM*5].detach().clone().cpu().numpy()

    plt.plot(np.mean(thal_activity[0, :, :], axis=-1))
    plt.plot(np.mean(thal_activity[1, :, :], axis=-1))
    plt.plot(np.mean(thal_activity[2, :, :], axis=-1))
    plt.plot(np.mean(thal_activity[3, :, :], axis=-1))
    plt.show()

    thal2str_acts = []
    for cond in range(iti_inp.shape[0]):
        thal2str_act_cur = np.mean((thal2str @ thal_activity[cond].T).T, axis=-1)
        thal2str_acts.append(thal2str_act_cur)
    
    plt.plot(thal2str_acts[0])
    plt.plot(thal2str_acts[1])
    plt.plot(thal2str_acts[2])
    plt.plot(thal2str_acts[3])
    plt.show()

    thal_acts_manipulation = []
    thal2str_acts_manipulation = []
    for cond in range(iti_inp.shape[0]):
        act_manipulation = get_acts_manipulation(
            len_seq, 
            rnn, 
            HID_DIM,
            INP_DIM,
            x_data,
            cond,
            MODEL_TYPE,
            ITI_STEPS,
            START_SILENCE,
            END_SILENCE,
            STIM_STRENGTH,
            EXTRA_STEPS,
            REGION_TO_SILENCE
            )

        thal_act_manipulation_cur = act_manipulation[:, HID_DIM*4:HID_DIM*5]
        thal2str_act_manipulation_cur = np.mean((thal2str @ thal_act_manipulation_cur.T).T, axis=-1)

        thal_acts_manipulation.append(thal_act_manipulation_cur)
        thal2str_acts_manipulation.append(thal2str_act_manipulation_cur)
    
    plt.plot(np.mean(thal_acts_manipulation[0], axis=-1))
    plt.plot(np.mean(thal_acts_manipulation[1], axis=-1))
    plt.plot(np.mean(thal_acts_manipulation[2], axis=-1))
    plt.plot(np.mean(thal_acts_manipulation[3], axis=-1))
    plt.show()

    plt.plot(thal2str_acts_manipulation[0])
    plt.plot(thal2str_acts_manipulation[1])
    plt.plot(thal2str_acts_manipulation[2])
    plt.plot(thal2str_acts_manipulation[3])
    plt.show()

if __name__ == "__main__":
    main()