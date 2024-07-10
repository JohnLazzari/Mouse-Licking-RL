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
CONSTRAINED = True
CHECK_PATH_D1D2 = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
CHECK_PATH_D1 = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1.pth"
CHECK_PATH_STRALM = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_stralm.pth"

def get_str2thal_weights_d1d2(rnn, constrained=True):
    
    if constrained == True:

        str2str = (rnn.str2str_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-15, 1) + rnn.str2str_fixed) @ rnn.str2str_D
        thal2str = rnn.thal2str_mask * F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-15, 1)
        str2snr = (rnn.str2snr_mask * F.hardtanh(rnn.str2snr_weight_l0_hh, 1e-15, 1)) @ rnn.str2snr_D
        str2gpe = (rnn.str2gpe_mask * F.hardtanh(rnn.str2gpe_weight_l0_hh, 1e-15, 1)) @ rnn.str2gpe_D
        gpe2stn = F.hardtanh(rnn.gpe2stn_weight_l0_hh, 1e-15, 1) @ rnn.gpe2stn_D
        stn2snr = F.hardtanh(rnn.stn2snr_weight_l0_hh, 1e-15, 1)
        snr2thal = F.hardtanh(rnn.snr2thal_weight_l0_hh, 1e-15, 1) @ rnn.snr2thal_D

    else:

        str2str = rnn.str2str_weight_l0_hh
        thal2str = rnn.thal2str_weight_l0_hh
        str2snr = rnn.str2snr_weight_l0_hh
        str2gpe = rnn.str2gpe_weight_l0_hh
        gpe2stn = rnn.gpe2stn_weight_l0_hh
        stn2snr = rnn.stn2snr_weight_l0_hh
        snr2thal = rnn.snr2thal_weight_l0_hh

    # Concatenate into single weight matrix

                        # STR       GPE         STN         SNR       Thal      ALM
    W_str = torch.cat([str2str, rnn.zeros, rnn.zeros, rnn.zeros, thal2str], dim=1)          # STR
    W_gpe = torch.cat([str2gpe, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # GPE
    W_stn = torch.cat([rnn.zeros, gpe2stn, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # STN
    W_snr = torch.cat([str2snr, rnn.zeros, stn2snr, rnn.zeros, rnn.zeros], dim=1)        # SNR
    W_thal = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, snr2thal, rnn.zeros], dim=1)   # Thal

    # Putting all weights together
    W_rec = torch.cat([W_str, W_gpe, W_stn, W_snr, W_thal], dim=0)
    
    return W_rec

def get_str2thal_weights_d1(rnn, constrained=True):
    
    if constrained == True:

        # Get full weights for training
        str2str = (rnn.str2str_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-15, 1) + rnn.str2str_fixed) @ rnn.str2str_D
        thal2str = rnn.thal2str_mask * F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-15, 1)
        str2snr = F.hardtanh(rnn.str2snr_weight_l0_hh, 1e-15, 1) @ rnn.str2snr_D
        snr2thal = F.hardtanh(rnn.snr2thal_weight_l0_hh, 1e-15, 1) @ rnn.snr2thal_D

    else:

        # Get full weights for training
        str2str = rnn.str2str_weight_l0_hh
        thal2str = rnn.thal2str_weight_l0_hh
        str2snr = rnn.str2snr_weight_l0_hh
        snr2thal = rnn.snr2thal_weight_l0_hh

    # Concatenate into single weight matrix

                        # STR       SNR       Thal    
    W_str = torch.cat([str2str, rnn.zeros, thal2str], dim=1)          # STR
    W_snr = torch.cat([str2snr, rnn.zeros, rnn.zeros], dim=1)        # SNR
    W_thal = torch.cat([rnn.zeros, snr2thal, rnn.zeros], dim=1)   # Thal

    # Putting all weights together
    W_rec = torch.cat([W_str, W_snr, W_thal], dim=0)

    return W_rec
    
def get_str2thal_weights_stralm(rnn, constrained=True):
    
    if constrained == True:

        str2str = (rnn.str2str_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-15, 1) + rnn.str2str_fixed) @ rnn.str2str_D
        str2alm = F.hardtanh(rnn.str2alm_weight_l0_hh, 1e-15, 1)
        alm2alm = F.hardtanh(rnn.alm2alm_weight_l0_hh, 1e-15, 1) @ rnn.alm2alm_D
        alm2str = rnn.alm2str_mask * F.hardtanh(rnn.alm2str_weight_l0_hh, 1e-15, 1)
    
    else:

        # Get full weights for training
        str2str = rnn.str2str_weight_l0_hh
        alm2alm = rnn.alm2alm_weight_l0_hh
        alm2str = rnn.alm2str_weight_l0_hh
        str2alm = rnn.str2str_weight_l0_hh
    
    # Concatenate into single weight matrix

                        # STR        ALM
    W_str = torch.cat([str2str, alm2str], dim=1)          # STR
    W_alm = torch.cat([str2alm, alm2alm], dim=1)       # ALM

    # Putting all weights together
    W_rec = torch.cat([W_str, W_alm], dim=0)

    return W_rec

def main():
    
    checkpoint_d1d2 = torch.load(CHECK_PATH_D1D2)
    checkpoint_d1 = torch.load(CHECK_PATH_D1)
    checkpoint_stralm = torch.load(CHECK_PATH_STRALM)
    
    # Values for D1D2 model
    rnn_d1d2 = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    rnn_d1d2.load_state_dict(checkpoint_d1d2)
    total_num_units_d1d2 = HID_DIM * 6
    str2thal_start_d1d2 = 0
    str2thal_end_d1d2 = HID_DIM * 5
    inp_mask_d1d2 = rnn_d1d2.strthal_mask

    # Values for D1 model
    rnn_d1 = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    rnn_d1.load_state_dict(checkpoint_d1)
    total_num_units_d1 = HID_DIM * 4
    str2thal_start_d1 = 0
    str2thal_end_d1 = HID_DIM * 3
    inp_mask_d1 = rnn_d1.strthal_mask

    # Values for STRALM model
    rnn_stralm = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    rnn_stralm.load_state_dict(checkpoint_stralm)
    total_num_units_stralm = HID_DIM * 2
    str2thal_start_stralm = 0
    str2thal_end_stralm = HID_DIM
    inp_mask_stralm = rnn_stralm.str_d1_mask

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    
    str2thal_weight_d1d2 = get_str2thal_weights_d1d2(rnn_d1d2, constrained=CONSTRAINED) 
    str2thal_weight_d1d2 = str2thal_weight_d1d2.detach().cpu().numpy()

    str2thal_weight_d1 = get_str2thal_weights_d1(rnn_d1, constrained=CONSTRAINED) 
    str2thal_weight_d1 = str2thal_weight_d1.detach().cpu().numpy()

    str2alm_weight = get_str2thal_weights_stralm(rnn_stralm, constrained=CONSTRAINED) 
    str2alm_weight = str2alm_weight.detach().cpu().numpy()

    eigenvalues_d1d2, eigenvectors_d1d2 = np.linalg.eig(str2thal_weight_d1d2.T)
    left_unitary_eigenvector_d1d2 = eigenvectors_d1d2[np.argmax(eigenvalues_d1d2)]

    eigenvalues_d1, eigenvectors_d1 = np.linalg.eig(str2thal_weight_d1.T)
    left_unitary_eigenvector_d1 = eigenvectors_d1[np.argmax(eigenvalues_d1)]

    eigenvalues_stralm, eigenvectors_stralm = np.linalg.eig(str2alm_weight.T)
    left_unitary_eigenvector_stralm = eigenvectors_stralm[np.argmax(eigenvalues_stralm)]

    plt.plot(np.linspace(0, eigenvalues_d1d2.shape[0], eigenvalues_d1d2.shape[0]), np.sort(np.real(eigenvalues_d1d2))[::-1], linewidth=10)
    plt.plot(np.linspace(100, eigenvalues_d1.shape[0] + 100, eigenvalues_d1.shape[0]), np.sort(np.real(eigenvalues_d1))[::-1], linewidth=10)
    plt.plot(np.linspace(200, eigenvalues_stralm.shape[0] + 200, eigenvalues_stralm.shape[0]), np.sort(np.real(eigenvalues_stralm))[::-1], linewidth=10)
    plt.ylim(top=1, bottom=-0.6)
    plt.xticks([])
    plt.show()

    str2thal_inp = (x_data @ F.hardtanh(rnn.inp_weight, 1e-15, 1) * inp_mask) + rnn.tonic_inp
    str2thal_inp = str2thal_inp[2, 1001, str2thal_start:str2thal_end].detach().cpu().numpy()

    inner_product = np.inner(left_unitary_eigenvector, str2thal_inp)
    print(inner_product)


    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()

    # Inhibitory/excitatory stimulus to network, designed as an input current
    # Does this for a single condition, len_seq should be a single number for the chosen condition, and x_data should be [1, len_seq, :]
    inhib_stim_pre = 0 * torch.ones(size=(3, 1000, total_num_units), device="cuda") * rnn.alm_mask
    inhib_stim_silence = 0 * torch.ones(size=(3, 3100 - 1000, total_num_units), device="cuda") * rnn.alm_mask
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