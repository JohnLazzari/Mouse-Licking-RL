import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_inp_data, get_acts_control, get_acts_manipulation, get_data
import tqdm
import time

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 16}
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.linewidth'] = 4 # set the value globally
plt.rc('font', **font)

HID_DIM = 256
OUT_DIM = 1451
INP_DIM = int(HID_DIM*0.1)
DT = 1e-2
CONDS = 4
MODEL_TYPE = "d1d2" # d1d2, d1, stralm
CHECK_PATH = f"checkpoints/{MODEL_TYPE}_datadriven_full_simulated_256n_nonoise_10000iters.pth"
SAVE_NAME_PATH = f"results/multi_regional_perturbations/{MODEL_TYPE}/"
INP_PATH = "data/firing_rates/ITIProj_trialPlotAll1.mat"
CONSTRAINED = True
ITI_STEPS = 100
START_SILENCE = 160                    # timepoint from start of trial to silence at
END_SILENCE = 220                      # timepoint from start of trial to end silencing
EXTRA_STEPS_SILENCE = 100
EXTRA_STEPS_CONTROL = 0
SILENCED_REGION = "alm"
STIM_STRENGTH = 5
PCA = False
TRIAL_EPOCH = "full"
INP_TYPE = "simulated"

def plot_psths(
            len_seq, 
            rnn, 
            iti_inp, 
            cue_inp, 
            peaks,
            silence=False
        ):
    
    if silence:

        # activity without silencing
        act_conds = get_acts_manipulation(
            len_seq, 
            rnn, 
            HID_DIM, 
            INP_DIM,
            MODEL_TYPE, 
            START_SILENCE,
            END_SILENCE,
            STIM_STRENGTH, 
            SILENCED_REGION,
            DT,
            TRIAL_EPOCH,
            peaks,
            INP_TYPE
        )
    
    else:

        # activity without silencing
        act_conds = get_acts_control(
            len_seq, 
            rnn, 
            HID_DIM, 
            INP_DIM,
            iti_inp,
            cue_inp,
            MODEL_TYPE
        )

    '''
        plt.plot(act_conds[0, :, HID_DIM * 5 + fsi_size:HID_DIM * 6], linewidth=6)
        plt.show()
    '''
    

    plt.plot(act_conds[0, :, :HID_DIM], linewidth=6)
    plt.show()

    fig, axs = plt.subplots(2, 3)

    axs[0, 0].plot(np.mean(act_conds[:, :, :int(HID_DIM/2)], axis=-1).T, linewidth=6)
    axs[0, 0].set_title("D1 PSTH")

    axs[0, 1].plot(np.mean(act_conds[:, :, int(HID_DIM/2):HID_DIM], axis=-1).T, linewidth=6)
    axs[0, 1].set_title("D2 PSTH")

    axs[0, 2].plot(np.mean(act_conds[:, :, HID_DIM:int(HID_DIM/2) + HID_DIM], axis=-1).T, linewidth=6)
    axs[0, 2].set_title("STN PSTH")

    axs[1, 0].plot(np.mean(act_conds[:, :, int(HID_DIM/2) + HID_DIM:int(HID_DIM/2) + HID_DIM * 2], axis=-1).T, linewidth=6)
    axs[1, 0].set_title("Thal PSTH")

    axs[1, 1].plot(np.mean(act_conds[:, :, int(HID_DIM/2) + HID_DIM * 2:int(HID_DIM/2) + HID_DIM * 3 - int(HID_DIM * 0.3)], axis=-1).T, linewidth=6)
    axs[1, 1].set_title("ALM Excitatory PSTH")

    axs[1, 2].plot(np.mean(act_conds[:, :, int(HID_DIM/2) + HID_DIM * 3 - int(HID_DIM * 0.3):int(HID_DIM/2) + HID_DIM * 3], axis=-1).T, linewidth=6)
    axs[1, 2].set_title("ALM Inhibitory PSTH")

    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "d1d2":

        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()

    elif MODEL_TYPE == "stralm":

        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()

    rnn.load_state_dict(checkpoint)

    # Get ramping activity
    neural_act, peak_times = get_data(DT, TRIAL_EPOCH, pca=PCA)
    neural_act = neural_act.cuda()

    iti_inp, cue_inp, len_seq = gather_inp_data(DT, HID_DIM, INP_PATH, TRIAL_EPOCH, peak_times, inp_type=INP_TYPE)
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()
    
    plot_psths(
        len_seq, 
        rnn, 
        iti_inp, 
        cue_inp, 
        peak_times
    )

    plot_psths(
        len_seq, 
        rnn, 
        iti_inp, 
        cue_inp, 
        peak_times,
        silence=True
    )

if __name__ == "__main__":
    main()