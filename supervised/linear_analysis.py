import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy
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

def analysis(rnn, str2thal_weight, x_data, len_seq, inp_mask, str2thal_start, str2thal_end, alm_start, alm_end, total_num_units):

    eigenvalues, eigenvectors = np.linalg.eig(str2thal_weight.T)
    left_unitary_eigenvector = eigenvectors[np.argmax(eigenvalues)]

    str2thal_inp = (x_data @ F.hardtanh(rnn.inp_weight, 1e-15, 1) * inp_mask)
    str2thal_inp = str2thal_inp[:, :, str2thal_start:str2thal_end].detach().cpu().numpy()

    inner_product = np.inner(str2thal_inp, np.real(left_unitary_eigenvector))

    hn = torch.zeros(size=(1, 3, total_num_units), device="cuda")
    inhib_stim = torch.zeros(size=(3, x_data.shape[1], total_num_units), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, act = rnn(x_data, hn, inhib_stim, noise=False)
    
    alm2str_weights = rnn.alm2str_weight_l0_hh.detach().cpu().numpy()
    alm_act = np.transpose(act[:, :, alm_start:alm_end].detach().cpu().numpy(), axes=(2, 1, 0))

    alm_act_conds = []
    for cond in range(x_data.shape[0]):
        alm_act_conds.append(alm2str_weights @ alm_act[:, :, cond])

    ramp_inp = np.stack(alm_act_conds, axis=2)
    ramp_inp = np.transpose(ramp_inp, axes=(2, 1, 0))

    mask = np.zeros(shape=(ramp_inp.shape[0], ramp_inp.shape[1], left_unitary_eigenvector.shape[0] - ramp_inp.shape[2]))
    ramp_inp = np.concatenate([ramp_inp, mask], axis=2)

    inner_product_alm = np.inner(ramp_inp, np.real(left_unitary_eigenvector))

    plt.plot(inner_product[0, :len_seq[0]], linewidth=10)
    plt.plot(inner_product[1, :len_seq[1]], linewidth=10)
    plt.plot(inner_product[2, :len_seq[2]], linewidth=10)
    plt.xticks([])
    plt.show()

    plt.plot(inner_product_alm[0, :len_seq[0]], linewidth=10)
    plt.plot(inner_product_alm[1, :len_seq[1]], linewidth=10)
    plt.plot(inner_product_alm[2, :len_seq[2]], linewidth=10)
    plt.xticks([])
    plt.show()

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
    alm_start_d1d2 = HID_DIM*5
    alm_end_d1d2 = HID_DIM*6
    inp_mask_d1d2 = rnn_d1d2.strthal_mask

    # Values for D1 model
    rnn_d1 = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    rnn_d1.load_state_dict(checkpoint_d1)
    total_num_units_d1 = HID_DIM * 4
    str2thal_start_d1 = 0
    str2thal_end_d1 = HID_DIM * 3
    alm_start_d1 = HID_DIM*3
    alm_end_d1 = HID_DIM*4
    inp_mask_d1 = rnn_d1.strthal_mask

    # Values for STRALM model
    rnn_stralm = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    rnn_stralm.load_state_dict(checkpoint_stralm)
    total_num_units_stralm = HID_DIM * 2
    str2thal_start_stralm = 0
    str2thal_end_stralm = HID_DIM * 2
    alm_start_stralm = HID_DIM
    alm_end_stralm = HID_DIM*2
    inp_mask_stralm = rnn_stralm.str_d1_mask

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    
    str2thal_weight_d1d2 = get_str2thal_weights_d1d2(rnn_d1d2, constrained=CONSTRAINED) 
    str2thal_weight_d1d2 = str2thal_weight_d1d2.detach().cpu().numpy().astype(np.float64)

    str2thal_weight_d1 = get_str2thal_weights_d1(rnn_d1, constrained=CONSTRAINED) 
    str2thal_weight_d1 = str2thal_weight_d1.detach().cpu().numpy().astype(np.float64)

    str2thal_weight_stralm = get_str2thal_weights_stralm(rnn_stralm, constrained=CONSTRAINED) 
    str2thal_weight_stralm = str2thal_weight_stralm.detach().cpu().numpy().astype(np.float64)

    analysis(rnn_d1d2, str2thal_weight_d1d2, x_data, len_seq, inp_mask_d1d2, str2thal_start_d1d2, str2thal_end_d1d2, alm_start_d1d2, alm_end_d1d2, total_num_units_d1d2)
    analysis(rnn_d1, str2thal_weight_d1, x_data, len_seq, inp_mask_d1, str2thal_start_d1, str2thal_end_d1, alm_start_d1, alm_end_d1, total_num_units_d1)
    analysis(rnn_stralm, str2thal_weight_stralm, x_data, len_seq, inp_mask_stralm, str2thal_start_stralm, str2thal_end_stralm, alm_start_stralm, alm_end_stralm, total_num_units_stralm)

if __name__ == "__main__":
    main()