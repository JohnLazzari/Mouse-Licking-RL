import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def loss_d1d2(constraint_criterion, 
                thresh_criterion,
                act, 
                neural_act_alm, 
                neural_act_str, 
                neural_act_thal, 
                hid_dim, 
                alm_start, 
                str_start, 
                thal_start,
                type="alm"):

    if type == "alm":

        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:alm_start + (hid_dim - int(0.3 * hid_dim))], dim=-1, keepdim=True), neural_act_alm[:, 500:, :])
                + 1e-3 * torch.mean(torch.pow(act[:, 1000:, :], 2), dim=(1, 2, 0))
                )

        '''
        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:alm_start + hid_dim], dim=-1, keepdim=True), neural_act_alm[:, 500:, :])
                + 1e-3 * torch.mean(torch.pow(act[:, 1000:, :], 2), dim=(1, 2, 0))  
                )
        '''
    
    elif type == "threshold":

        mean_act = torch.mean(act[:, :, alm_start:alm_start+hid_dim], dim=-1, keepdim=True)

        loss = (
                # event target should be only for delay (in terms of number of timesteps it has)
                constraint_criterion(mean_act[0, 2000-1, :], neural_act_alm[0, 2000-1, :])
                + constraint_criterion(mean_act[1, 2500-1, :], neural_act_alm[1, 2500-1, :])
                + constraint_criterion(mean_act[2, 3000-1, :], neural_act_alm[2, 3000-1, :])
                + constraint_criterion(mean_act[3, 3500-1, :], neural_act_alm[3, 3500-1, :])
                + constraint_criterion(mean_act[4, 4000-1, :], neural_act_alm[4, 4000-1, :])
                + constraint_criterion(mean_act[:, 500:1000, :], neural_act_alm[:, 500:1000, :])
                + 1e-1 * torch.mean(torch.pow(act[:, 1001:, :], 2), dim=(1, 2, 0))  
                )

    else:

        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:alm_start+hid_dim], dim=-1, keepdim=True), neural_act_alm[:, 500:, :])
                + 1e-3 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
                + constraint_criterion(torch.mean(act[:, 500:, str_start:str_start+hid_dim], dim=-1, keepdim=True), neural_act_str[:, 500:, :])
                + constraint_criterion(torch.mean(act[:, 500:, thal_start:thal_start+hid_dim], dim=-1, keepdim=True), neural_act_thal[:, 500:, :])
                )
    
    return loss

def loss_stralm(constraint_criterion, 
                act, 
                neural_act_alm, 
                neural_act_str, 
                alm_start, 
                str_start,
                type="alm"):

    if type == "alm":

        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:], dim=-1, keepdim=True), neural_act_alm[:, 500:, :])
                + 1e-4 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
                )
    
    else:

        loss = (
                constraint_criterion(torch.mean(act[:, 500:, alm_start:], dim=-1, keepdim=True), neural_act_alm[:, 500:, :])
                + 1e-4 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
                + constraint_criterion(torch.mean(act[:, 500:, str_start:alm_start], dim=-1, keepdim=True), neural_act_str[:, 500:, :])
                )
    
    return loss

def simple_dynamics_d1d2(act, rnn, hid_dim):

    # Get full weights for training
    str2str = (rnn.str2str_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
    alm2alm = F.hardtanh(rnn.alm2alm_weight_l0_hh, 1e-10, 1) @ rnn.alm2alm_D
    alm2str = rnn.alm2str_mask * F.hardtanh(rnn.alm2str_weight_l0_hh, 1e-10, 1)
    alm2thal = rnn.alm2thal_mask * F.hardtanh(rnn.alm2thal_weight_l0_hh, 1e-10, 1)
    thal2alm = F.hardtanh(rnn.thal2alm_weight_l0_hh, 1e-10, 1)
    thal2str = F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-10, 1)
    str2snr = (rnn.str2snr_mask * F.hardtanh(rnn.str2snr_weight_l0_hh, 1e-10, 1)) @ rnn.str2snr_D
    str2gpe = (rnn.str2gpe_mask * F.hardtanh(rnn.str2gpe_weight_l0_hh, 1e-10, 1)) @ rnn.str2gpe_D
    gpe2stn = F.hardtanh(rnn.gpe2stn_weight_l0_hh, 1e-10, 1) @ rnn.gpe2stn_D
    stn2snr = F.hardtanh(rnn.stn2snr_weight_l0_hh, 1e-10, 1)
    snr2thal = F.hardtanh(rnn.snr2thal_weight_l0_hh, 1e-10, 1) @ rnn.snr2thal_D

    # Concatenate into single weight matrix

                        # STR       GPE         STN         SNR       Thal      ALM
    W_str = torch.cat([str2str, rnn.zeros, rnn.zeros, rnn.zeros, thal2str, alm2str], dim=1)          # STR
    W_gpe = torch.cat([str2gpe, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # GPE
    W_stn = torch.cat([rnn.zeros, gpe2stn, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # STN
    W_snr = torch.cat([str2snr, rnn.zeros, stn2snr, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)        # SNR
    W_thal = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, snr2thal, rnn.zeros, rnn.zeros], dim=1)   # Thal
    W_alm = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, thal2alm, alm2alm], dim=1)       # ALM

    # Putting all weights together
    W_rec = torch.cat([W_str, W_gpe, W_stn, W_snr, W_thal, W_alm], dim=0)

    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act[:, 1001:, :] > 0, 1., 0.), dim=(1, 0))

    update = 1e-1 * W_rec * d_act[:hid_dim*6]

    rnn.str2str_weight_l0_hh.grad += update[:hid_dim, :hid_dim]
    rnn.thal2str_weight_l0_hh.grad += update[:hid_dim, hid_dim*4:hid_dim*5]
    rnn.alm2str_weight_l0_hh.grad += update[:hid_dim, hid_dim*5:hid_dim*6]
    rnn.str2gpe_weight_l0_hh.grad += update[hid_dim:hid_dim*2, :hid_dim]
    rnn.gpe2stn_weight_l0_hh.grad += update[hid_dim*2:hid_dim*3, hid_dim:hid_dim*2]
    rnn.str2snr_weight_l0_hh.grad += update[hid_dim*3:hid_dim*4, :hid_dim]
    rnn.stn2snr_weight_l0_hh.grad += update[hid_dim*3:hid_dim*4, hid_dim*2:hid_dim*3]
    rnn.snr2thal_weight_l0_hh.grad += update[hid_dim*4:hid_dim*5, hid_dim*3:hid_dim*4]
    rnn.thal2alm_weight_l0_hh.grad += update[hid_dim*5:hid_dim*6, hid_dim*4:hid_dim*5]
    rnn.alm2alm_weight_l0_hh.grad += update[hid_dim*5:hid_dim*6, hid_dim*5:hid_dim*6]
    #rnn.alm2thal_weight_l0_hh.grad += update[hid_dim*4:hid_dim*5, hid_dim*5:hid_dim*6]
