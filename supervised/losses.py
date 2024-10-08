import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def loss_d1d2(
    rnn,
    constraint_criterion, 
    act, 
    neural_act, 
    alm_start,
    alm_end
):
    
    loss = (
            constraint_criterion(act[:, :, :], neural_act[:, :, :])
            )
    
    return loss

def loss_stralm(constraint_criterion, 
                act, 
                neural_act, 
                alm_start, 
                str_start,
                type="alm"):

    if type == "alm":

        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:], dim=-1, keepdim=True), neural_act[:, 500:, :])
                + 1e-4 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
                )
    
    else:

        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:], dim=-1, keepdim=True), neural_act[:, 500:, :])
                + 1e-4 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
                + constraint_criterion(torch.mean(act[:, 500:, str_start:alm_start], dim=-1, keepdim=True), neural_act[:, 500:, :])
                )
    
    return loss

def simple_dynamics_d1d2(act, rnn, hid_dim):

    # Get full weights for training
    d12d1 = (rnn.str2str_sparse_mask * F.hardtanh(rnn.d12d1_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
    d12d2 = (rnn.str2str_sparse_mask * F.hardtanh(rnn.d12d2_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
    d22d1 = (rnn.str2str_sparse_mask * F.hardtanh(rnn.d22d1_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
    d22d2 = (rnn.str2str_sparse_mask * F.hardtanh(rnn.d22d2_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
    d22stn = F.hardtanh(rnn.d22stn_weight_l0_hh, 1e-10, 1)
    d12thal = F.hardtanh(rnn.d12thal_weight_l0_hh, 1e-10, 1)
    alm2alm = F.hardtanh(rnn.alm2alm_weight_l0_hh, 1e-10, 1) @ rnn.alm2alm_D
    alm2d1 = rnn.alm2str_mask * F.hardtanh(rnn.alm2d1_weight_l0_hh, 1e-10, 1)
    alm2d2 = rnn.alm2str_mask * F.hardtanh(rnn.alm2d2_weight_l0_hh, 1e-10, 1)
    thal2alm = F.hardtanh(rnn.thal2alm_weight_l0_hh, 1e-10, 1)
    thal2d1 = F.hardtanh(rnn.thal2d1_weight_l0_hh, 1e-10, 1)
    thal2d2 = F.hardtanh(rnn.thal2d2_weight_l0_hh, 1e-10, 1)
    stn2thal = F.hardtanh(rnn.stn2thal_weight_l0_hh, 1e-10, 1) @ rnn.stn2thal_D
    inp_weight_d1 = F.hardtanh(rnn.inp_weight_d1, 1e-10, 1)
    inp_weight_d2 = F.hardtanh(rnn.inp_weight_d2, 1e-10, 1)
    out_weight_alm = F.hardtanh(rnn.out_weight_alm, 1e-10, 1)

    # Concatenate into single weight matrix

                        # D1   D2     STN       Thal       ALM    
    W_d1 = torch.cat([d12d1, d22d1, rnn.small_to_small, thal2d1, alm2d1, inp_weight_d1],                     dim=1) # D1
    W_d2 = torch.cat([d12d2, d22d2, rnn.small_to_small, thal2d2, alm2d2, inp_weight_d2],                     dim=1) # D2
    W_stn = torch.cat([rnn.small_to_small, d22stn, rnn.small_to_small, rnn.large_to_small, rnn.large_to_small, rnn.zeros_from_iti_small],                dim=1) # STN
    W_thal = torch.cat([d12thal, rnn.small_to_large, stn2thal, rnn.large_to_large, rnn.large_to_large, rnn.zeros_from_iti_large],                dim=1) # Thal
    W_alm = torch.cat([rnn.small_to_large, rnn.small_to_large, rnn.small_to_large, thal2alm, alm2alm, rnn.zeros_from_iti_large],                  dim=1) # ALM
    W_iti = torch.cat([rnn.zeros_to_iti_small, rnn.zeros_to_iti_small, rnn.zeros_to_iti_small, rnn.zeros_to_iti_large, rnn.zeros_to_iti_large, rnn.zeros_rec_iti], dim=1)

    # Putting all weights together
    W_rec = torch.cat([W_d1, W_d2, W_stn, W_thal, W_alm, W_iti], dim=0)

    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

    update = 1e-4 * W_rec * d_act

    update = torch.mean(update)

    return update