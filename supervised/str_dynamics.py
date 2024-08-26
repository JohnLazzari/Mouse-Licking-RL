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
DT = 1e-2
CONDITION = 0
START_SILENCE = 160
END_SILENCE = 220
MODEL_TYPE = "d1d2"
STIM_STRENGTH = 10
REGION_TO_SILENCE = "alm"
EXTRA_STEPS = 100
ITI_STEPS = 100
CHECK_PATH = f"checkpoints/{MODEL_TYPE}_fsi2str_256n_almnoise.1_itinoise.05_10000iters_newloss.pth"

def main():
    
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)

    total_num_units = HID_DIM * 6 + INP_DIM + int(HID_DIM * 0.3)
    str_start = 0
    stn_start = HID_DIM*2
    snr_start = HID_DIM*3

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    iti_inp, cue_inp = x_data
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Sample many hidden states to get pcs for dimensionality reduction
    hn = torch.zeros(size=(1, 4, total_num_units)).cuda()
    xn = torch.zeros(size=(1, 4, total_num_units)).cuda()

    inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, _, act = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=False)
    
    thal2str = F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-10, 1).detach().cpu().numpy()
    alm2str = (rnn.alm2str_mask * F.hardtanh(rnn.alm2str_weight_l0_hh, 1e-10, 1)).detach().cpu().numpy()
    inp_weight_str = F.hardtanh(rnn.inp_weight_str, 1e-10, 1).detach().cpu().numpy()
    str2str = ((rnn.str2str_sparse_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D).detach().cpu().numpy()

    thal_activity = act[:, :, HID_DIM*4:HID_DIM*5].detach().clone().cpu().numpy()
    alm_activity = act[:, :, HID_DIM*5:HID_DIM*6].detach().clone().cpu().numpy()
    iti_activity = act[:, :, HID_DIM*6:HID_DIM*6 + INP_DIM].detach().clone().cpu().numpy()
    str_activity = act[:, :, :HID_DIM].detach().clone().cpu().numpy()

    thal2str_acts = []
    alm2str_acts = []
    iti2str_acts = []
    str_decay = []

    for cond in range(iti_inp.shape[0]):

        thal2str_act_cur = np.mean((thal2str @ thal_activity[cond].T).T, axis=-1)
        alm2str_act_cur = np.mean((alm2str @ alm_activity[cond].T).T, axis=-1)
        iti2str_act_cur = np.mean((inp_weight_str @ iti_activity[cond].T).T, axis=-1)
        str_decay_cur = np.mean((str2str @ str_activity[cond].T).T, axis=-1) - np.mean(str_activity[cond], axis=-1)

        thal2str_acts.append(thal2str_act_cur)
        alm2str_acts.append(alm2str_act_cur)
        iti2str_acts.append(iti2str_act_cur)
        str_decay.append(str_decay_cur)
    
    plt.plot(np.array(thal2str_acts).T, linewidth=6)
    plt.show()

    plt.plot(np.array(alm2str_acts).T, linewidth=6)
    plt.show()

    plt.plot(np.array(iti2str_acts).T, linewidth=6)
    plt.show()

    plt.plot(np.array(str_decay).T, linewidth=6)
    plt.show()

    thal2str_acts_manipulation = []
    alm2str_acts_manipulation = []
    iti2str_acts_manipulation = []
    str_decay_manipulation = []

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
        alm_act_manipulation_cur = act_manipulation[:, HID_DIM*5:HID_DIM*6]
        iti_act_manipulation_cur = act_manipulation[:, HID_DIM*6:HID_DIM*6+INP_DIM]
        str_act_manipulation_cur = act_manipulation[:, :HID_DIM]

        thal2str_act_manipulation_cur = np.mean((thal2str @ thal_act_manipulation_cur.T).T, axis=-1)
        alm2str_act_manipulation_cur = np.mean((alm2str @ alm_act_manipulation_cur.T).T, axis=-1)
        iti2str_act_manipulation_cur = np.mean((inp_weight_str @ iti_act_manipulation_cur.T).T, axis=-1)
        str_decay_manipulation_cur = np.mean((str2str @ thal_act_manipulation_cur.T).T, axis=-1) - np.mean(str_act_manipulation_cur, axis=-1)

        thal2str_acts_manipulation.append(thal2str_act_manipulation_cur)
        alm2str_acts_manipulation.append(alm2str_act_manipulation_cur)
        iti2str_acts_manipulation.append(iti2str_act_manipulation_cur)
        str_decay_manipulation.append(str_decay_manipulation_cur)
    
    plt.plot(thal2str_acts_manipulation[0], linewidth=6)
    plt.plot(thal2str_acts_manipulation[1], linewidth=6)
    plt.plot(thal2str_acts_manipulation[2], linewidth=6)
    plt.plot(thal2str_acts_manipulation[3], linewidth=6)
    plt.plot(thal2str_acts_manipulation[4], linewidth=6)
    plt.show()

    plt.plot(alm2str_acts_manipulation[0], linewidth=6)
    plt.plot(alm2str_acts_manipulation[1], linewidth=6)
    plt.plot(alm2str_acts_manipulation[2], linewidth=6)
    plt.plot(alm2str_acts_manipulation[3], linewidth=6)
    plt.plot(alm2str_acts_manipulation[4], linewidth=6)
    plt.show()

    plt.plot(iti2str_acts_manipulation[0], linewidth=6)
    plt.plot(iti2str_acts_manipulation[1], linewidth=6)
    plt.plot(iti2str_acts_manipulation[2], linewidth=6)
    plt.plot(iti2str_acts_manipulation[3], linewidth=6)
    plt.plot(iti2str_acts_manipulation[4], linewidth=6)
    plt.show()

    plt.plot(str_decay_manipulation[0], linewidth=6)
    plt.plot(str_decay_manipulation[1], linewidth=6)
    plt.plot(str_decay_manipulation[2], linewidth=6)
    plt.plot(str_decay_manipulation[3], linewidth=6)
    plt.plot(str_decay_manipulation[4], linewidth=6)
    plt.show()

if __name__ == "__main__":
    main()