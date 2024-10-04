import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM, RNN_MultiRegional_D1
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_inp_data, get_acts_control, get_acts_manipulation, get_ramp
import tqdm
import time

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 12}
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.linewidth'] = 4 # set the value globally
plt.rc('font', **font)

HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM * 0.1)                                                          
DT = 1e-2
CONDS = 4
MODEL_TYPE = "d1d2" # d1d2, d1, stralm
CHECK_PATH = f"checkpoints/{MODEL_TYPE}_full_256n_nonoise_10000iters_newloss.pth"
SAVE_NAME_PATH = f"results/multi_regional_perturbations/{MODEL_TYPE}/"
CONSTRAINED = True
ITI_STEPS = 100
START_SILENCE = 160                    # timepoint from start of trial to silence at
END_SILENCE = 220                      # timepoint from start of trial to end silencing
STIM_STRENGTH = 5
EXTRA_STEPS_SILENCE = 100
SILENCED_REGION = "alm"

def plot_psths(
            len_seq, 
            rnn, 
            x_data, 
            type="control"
        ):
    
    fsi_size = int(HID_DIM * 0.3)

    if type == "control":

        # activity without silencing
        act_conds = get_acts_control(
            len_seq, 
            rnn, 
            HID_DIM, 
            INP_DIM,
            x_data, 
            MODEL_TYPE
        )

    elif type == "manipulation": 

        # activity with silencing
        act_conds = get_acts_manipulation(
            len_seq, 
            rnn, 
            HID_DIM, 
            INP_DIM,
            MODEL_TYPE, 
            START_SILENCE,
            END_SILENCE,
            STIM_STRENGTH, 
            EXTRA_STEPS_SILENCE, 
            SILENCED_REGION,
            DT 
        )
        
    plt.plot(act_conds[0, :, HID_DIM * 5 + fsi_size:HID_DIM * 6 + fsi_size], linewidth=6)
    plt.show()
        
    fig, axs = plt.subplots(2, 5)

    axs[0, 0].plot(np.mean(act_conds[:, 50:, :HID_DIM], axis=-1).T, linewidth=6)
    axs[0, 0].set_title("D1 PSTH")

    axs[0, 1].plot(np.mean(act_conds[:, 50:, HID_DIM:HID_DIM*2], axis=-1).T, linewidth=6)
    axs[0, 1].set_title("D2 PSTH")

    axs[0, 2].plot(np.mean(act_conds[:, 50:, HID_DIM*2:HID_DIM*2 + fsi_size], axis=-1).T, linewidth=6)
    axs[0, 2].set_title("FSI PSTH")

    axs[0, 3].plot(np.mean(act_conds[:, 50:, HID_DIM*2 + fsi_size:HID_DIM * 3 + fsi_size], axis=-1).T, linewidth=6)
    axs[0, 3].set_title("GPe PSTH")

    axs[0, 4].plot(np.mean(act_conds[:, 50:, HID_DIM * 3 + fsi_size:HID_DIM * 4 + fsi_size], axis=-1).T, linewidth=6)
    axs[0, 4].set_title("STN PSTH")

    axs[1, 0].plot(np.mean(act_conds[:, 50:, HID_DIM * 4 + fsi_size:HID_DIM * 5 + fsi_size], axis=-1).T, linewidth=6)
    axs[1, 0].set_title("SNr PSTH")

    axs[1, 1].plot(np.mean(act_conds[:, 50:, HID_DIM * 5 + fsi_size:HID_DIM * 6 + fsi_size], axis=-1).T, linewidth=6)
    axs[1, 1].set_title("Thal PSTH")

    axs[1, 2].plot(np.mean(act_conds[:, 50:, HID_DIM * 6 + fsi_size:HID_DIM * 7], axis=-1).T, linewidth=6)
    axs[1, 2].set_title("ALM Excitatory PSTH")

    axs[1, 3].plot(np.mean(act_conds[:, 50:, HID_DIM * 7:HID_DIM * 7 + int(HID_DIM * 0.3)], axis=-1).T, linewidth=6)
    axs[1, 3].set_title("ALM Inhibitory PSTH")

    axs[1, 4].plot(np.mean(act_conds[:, 50:, HID_DIM * 7 + int(HID_DIM * 0.3):HID_DIM * 7 + int(HID_DIM * 0.3) + INP_DIM], axis=-1).T, linewidth=6)
    axs[1, 4].set_title("ITI PSTH")

    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "d1d2":

        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()

    elif MODEL_TYPE == "stralm":

        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()

    elif MODEL_TYPE == "d1":

        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()

    rnn.load_state_dict(checkpoint)

    # Get ramping activity
    neural_act = get_ramp(dt=DT)
    neural_act = neural_act.cuda()
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM, ramp=neural_act)
    
    plot_psths(
        len_seq, 
        rnn, 
        x_data,
         
    )

    plot_psths(
        len_seq, 
        rnn, 
        x_data, 
        type="manipulation"
    )

if __name__ == "__main__":
    main()