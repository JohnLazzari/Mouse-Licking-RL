import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_inp_data, get_acts_control, get_acts_manipulation

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.linewidth'] = 4 # set the value globally
font = {'size' : 20}
plt.rc('font', **font)

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
DT = 1e-3
ITI_STEPS = 1000
EXTRA_STEPS_CONTROL = 0
EXTRA_STEPS_MANIPULATION = 1000
START_SILENCE = 600 + ITI_STEPS
END_SILENCE = 1200 + ITI_STEPS 
CHECK_PATH = f"checkpoints/d1d2_256n_almnoise_newloss.pth"

def main():
    
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)

    # Get region start points
    str_mask = rnn.str_d2_mask
    total_num_units = HID_DIM * 6 + INP_DIM
    str_start = 0
    stn_start = HID_DIM*2
    snr_start = HID_DIM*3

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    iti_inp, cue_inp = x_data
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Init hidden state and inhib stim
    hn = torch.zeros(size=(1, 5, total_num_units)).cuda()
    inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, act, = rnn(iti_inp, cue_inp, hn, inhib_stim, noise=False)
    
    # Get SNr activity
    sampled_acts = act[:, 500:, :].detach().cpu().numpy()

    # Get weight matrices into SNr
    stn2snr = (F.hardtanh(rnn.stn2snr_weight_l0_hh, 1e-10, 1)).detach().cpu().numpy()
    str2snr = ((rnn.str2snr_mask * F.hardtanh(rnn.str2snr_weight_l0_hh, 1e-10, 1)) @ rnn.str2snr_D).detach().cpu().numpy()

    # Get the activity going into SNr from STN and STR
    act_conds_d1 = []
    act_conds_d2 = []
    for cond in range(5):

        d2_snr_activity = (stn2snr @ sampled_acts[cond, :, stn_start:stn_start+HID_DIM].T).T
        act_conds_d2.append(d2_snr_activity)

        d1_snr_activity = (str2snr @ sampled_acts[cond, :, str_start:str_start+HID_DIM].T).T
        act_conds_d1.append(d1_snr_activity)
    
    # Convert to numpy
    act_conds_d1 = np.array(act_conds_d1)
    act_conds_d2 = np.array(act_conds_d2)

    # Get the mean activity across neurons for each condition
    d2_snr_activity_reduced = np.mean(act_conds_d2, axis=-1)
    d1_snr_activity_reduced = np.mean(act_conds_d1, axis=-1)

    # X values for plotting
    x_0 = np.linspace(-0.5, (len_seq[0] - 1000) * DT, len_seq[0] - 500)
    x_1 = np.linspace(-0.5, (len_seq[1] - 1000) * DT, len_seq[1] - 500)
    x_2 = np.linspace(-0.5, (len_seq[2] - 1000) * DT, len_seq[2] - 500)
    x_3 = np.linspace(-0.5, (len_seq[3] - 1000) * DT, len_seq[3] - 500)
    x_4 = np.linspace(-0.5, (len_seq[4] - 1000) * DT, len_seq[4] - 500)

    plt.axvline(x=0.0, linestyle='--', color='black', label="Cue")

    plt.plot(x_0, d1_snr_activity_reduced[0, :len_seq[0] - 500], linewidth=6) 
    plt.plot(x_1, d1_snr_activity_reduced[1, :len_seq[1] - 500], linewidth=6) 
    plt.plot(x_2, d1_snr_activity_reduced[2, :len_seq[2] - 500], linewidth=6) 
    plt.plot(x_3, d1_snr_activity_reduced[3, :len_seq[3] - 500], linewidth=6) 
    plt.plot(x_4, d1_snr_activity_reduced[4, :len_seq[4] - 500], linewidth=6) 

    plt.plot(x_0, d2_snr_activity_reduced[0, :len_seq[0] - 500], linewidth=6) 
    plt.plot(x_1, d2_snr_activity_reduced[1, :len_seq[1] - 500], linewidth=6) 
    plt.plot(x_2, d2_snr_activity_reduced[2, :len_seq[2] - 500], linewidth=6) 
    plt.plot(x_3, d2_snr_activity_reduced[3, :len_seq[3] - 500], linewidth=6) 
    plt.plot(x_4, d2_snr_activity_reduced[4, :len_seq[4] - 500], linewidth=6) 

    plt.xlabel("Time (s)")
    plt.ylabel("Mean Activity")
    plt.show()

    # Get perturbed trajectories lists
    d1_perturbed_trajectories = []
    d2_perturbed_trajectories = []
    for cond in range(5):

        perturbed_act = get_acts_manipulation(
            len_seq,
            rnn, 
            HID_DIM, 
            INP_DIM,
            x_data,
            cond,
            "d1d2",
            ITI_STEPS,
            START_SILENCE,
            END_SILENCE,
            -10,
            EXTRA_STEPS_MANIPULATION,
            "alm"
        )

        d1_perturbed_trajectories.append(perturbed_act[500:len_seq[cond], str_start:str_start+HID_DIM])
        d2_perturbed_trajectories.append(perturbed_act[500:len_seq[cond], stn_start:stn_start+HID_DIM])

    projected_perturbed_d1 = []
    projected_perturbed_d2 = []

    for d1, d2 in zip(d1_perturbed_trajectories, d2_perturbed_trajectories):

        d1_snr_activity_perturbed = (str2snr @ d1.T).T
        projected_perturbed_d1.append(np.mean(d1_snr_activity_perturbed, axis=1))

        d2_snr_activity_perturbed = (stn2snr @ d2.T).T
        projected_perturbed_d2.append(np.mean(d2_snr_activity_perturbed, axis=1))
    
    for i in range(5):
        plt.plot(projected_perturbed_d1[i], linewidth=6) 
        plt.plot(projected_perturbed_d2[i], linewidth=6) 
    
    plt.show()

if __name__ == "__main__":
    main()