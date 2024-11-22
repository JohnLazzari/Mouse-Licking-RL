import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def loss(
    criterion, 
    out, 
    neural_act, 
):

    return criterion(out[:, 100:, :], neural_act[:, 100:, :])

    
def simple_dynamics_d1d2(act, rnn, hid_dim):

    fsi_size = int(hid_dim * 0.3)

    # Get full weights for training
    str2str = (rnn.str2str_sparse_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
    alm2alm = F.hardtanh(rnn.alm2alm_weight_l0_hh, 1e-10, 1) @ rnn.alm2alm_D
    alm2str = rnn.alm_ramp_2_d1_mask * rnn.alm2str_mask * F.hardtanh(rnn.alm2str_weight_l0_hh, 1e-10, 1)
    alm2thal = rnn.alm2thal_mask * F.hardtanh(rnn.alm2thal_weight_l0_hh, 1e-10, 1)
    thal2alm = F.hardtanh(rnn.thal2alm_weight_l0_hh, 1e-10, 1)
    thal2str = rnn.thal2str_mask * F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-10, 1)
    str2snr = (rnn.str2snr_mask * F.hardtanh(rnn.str2snr_weight_l0_hh, 1e-10, 1)) @ rnn.str2snr_D
    str2gpe = (rnn.str2gpe_mask * F.hardtanh(rnn.str2gpe_weight_l0_hh, 1e-10, 1)) @ rnn.str2gpe_D
    gpe2stn = F.hardtanh(rnn.gpe2stn_weight_l0_hh, 1e-10, 1) @ rnn.gpe2stn_D
    stn2snr = F.hardtanh(rnn.stn2snr_weight_l0_hh, 1e-10, 1)
    snr2thal = F.hardtanh(rnn.snr2thal_weight_l0_hh, 1e-10, 1) @ rnn.snr2thal_D
    fsi2str = F.hardtanh(rnn.fsi2str_weight, 1e-10, 1) @ rnn.fsi2str_D
    thal2fsi = rnn.thal2fsi_mask * F.hardtanh(rnn.thal2fsi_weight, 1e-10, 1)
    alm2fsi = rnn.alm_ramp_2_fsi_mask * rnn.alm2fsi_mask * F.hardtanh(rnn.alm2fsi_weight, 1e-10, 1)
    iti2fsi = rnn.iti_2_fsi_mask * F.hardtanh(rnn.iti2fsi_weight, 1e-10, 1)
    fsi2fsi = F.hardtanh(rnn.fsi2fsi_weight, 1e-10, 1) @ rnn.fsi2fsi_D
    inp_weight_str = rnn.iti_2_d1_mask * F.hardtanh(rnn.inp_weight_str, 1e-10, 1)

    # Concatenate into single weight matrix

                        # STR       GPE         STN         SNR       Thal      ALM         ALM ITI
    W_str = torch.cat([str2str, fsi2str, rnn.zeros, rnn.zeros, rnn.zeros, thal2str, alm2str], dim=1)                             # STR
    W_fsi = torch.cat([rnn.zeros_to_fsi, fsi2fsi, rnn.zeros_to_fsi, rnn.zeros_to_fsi, rnn.zeros_to_fsi, thal2fsi, alm2fsi], dim=1)     # FSI
    W_gpe = torch.cat([str2gpe, rnn.zeros_from_fsi, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)       # GPE
    W_stn = torch.cat([rnn.zeros, rnn.zeros_from_fsi, gpe2stn, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)       # STN
    W_snr = torch.cat([str2snr, rnn.zeros_from_fsi, rnn.zeros, stn2snr, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)          # SNR
    W_thal = torch.cat([rnn.zeros, rnn.zeros_from_fsi, rnn.zeros, rnn.zeros, snr2thal, rnn.zeros, rnn.zeros], dim=1)     # Thal
    W_alm = torch.cat([rnn.zeros, rnn.zeros_from_fsi, rnn.zeros, rnn.zeros, rnn.zeros, thal2alm, alm2alm], dim=1)         # ALM

    # Putting all weights together
    W_rec = torch.cat([W_str, W_fsi, W_gpe, W_stn, W_snr, W_thal, W_alm], dim=0)

    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act[:, 100:, :] > 0, 1., 0.), dim=(1, 0))

    update = 1e-3 * W_rec * d_act[:hid_dim*6 + fsi_size]
    update = torch.mean(update)

    return update