import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_delay_data, get_ramp

def loss_d1d2(criterion, 
                    constraint_criterion, 
                    act, 
                    out, 
                    neural_act, 
                    y_data, 
                    hid_dim, 
                    alm_start, 
                    str_start, 
                    thal_start):

    loss_stabilize = 1e-3 * torch.mean(torch.pow(act[:, :500, :], 2), dim=(1, 2, 0))

    loss_delay = (criterion(out[:, 500:, :], y_data[:, 500:, :]) 
            + constraint_criterion(torch.mean(act[:, 500:, alm_start:alm_start+hid_dim], dim=-1, keepdim=True), neural_act[:, 500:, :])
            + 1e-3 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
            + constraint_criterion(torch.mean(act[:, 500:, str_start:str_start+hid_dim], dim=-1, keepdim=True), neural_act[:, 500:, :])
            + constraint_criterion(torch.mean(act[:, 500:, thal_start:thal_start+hid_dim], dim=-1, keepdim=True), neural_act[:, 500:, :])
            )
    
    loss = loss_stabilize + loss_delay

    return loss

def loss_stralm(criterion, constraint_criterion, act, out, neural_act, y_data, alm_start, str_start):

    loss_stabilize = 1e-3 * torch.mean(torch.pow(act[:, :500, :], 2), dim=(1, 2, 0))
    
    loss_delay = (criterion(out[:, 500:, :], y_data[:, 500:, :]) 
                    + constraint_criterion(torch.mean(act[:, 500:, alm_start:], dim=-1, keepdim=True), neural_act[:, 500:, :])
                    + 1e-4 * torch.mean(torch.pow(act[:, 500:, :], 2), dim=(1, 2, 0))  
                    + constraint_criterion(torch.mean(act[:, 500:, str_start:alm_start], dim=-1, keepdim=True), neural_act[:, 500:, :])
                    )
    
    loss = loss_stabilize + loss_delay

    return loss

def simple_dynamics_d1d2(act, rnn, hid_dim):

    # Get full weights for training
    alm2alm = rnn.alm2alm_weight_l0_hh
    alm2str = rnn.alm2str_weight_l0_hh
    thal2alm = rnn.thal2alm_weight_l0_hh
    thal2str = rnn.thal2str_weight_l0_hh
    str2snr = rnn.str2snr_weight_l0_hh
    str2gpe = rnn.str2gpe_weight_l0_hh
    gpe2stn = rnn.gpe2stn_weight_l0_hh
    stn2snr = rnn.stn2snr_weight_l0_hh
    snr2thal = rnn.snr2thal_weight_l0_hh

    # Concatenate into single weight matrix

                        # STR       GPE         STN         SNR       Thal      ALM
    W_str = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, thal2str, alm2str], dim=1)          # STR
    W_gpe = torch.cat([str2gpe, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # GPE
    W_stn = torch.cat([rnn.zeros, gpe2stn, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # STN
    W_snr = torch.cat([str2snr, rnn.zeros, stn2snr, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)        # SNR
    W_thal = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, snr2thal, rnn.zeros, rnn.zeros], dim=1)   # Thal
    W_alm = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, thal2alm, alm2alm], dim=1)       # ALM

    # Putting all weights together
    W_rec = torch.cat([W_str, W_gpe, W_stn, W_snr, W_thal, W_alm], dim=0)

    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

    update = 1e-3 * W_rec.T * d_act

    rnn.thal2str_weight_l0_hh.grad += update[:hid_dim, hid_dim*4:hid_dim*5]
    rnn.alm2str_weight_l0_hh.grad += update[:hid_dim, hid_dim*5:hid_dim*6]
    rnn.str2gpe_weight_l0_hh.grad += update[hid_dim:hid_dim*2, :hid_dim]
    rnn.gpe2stn_weight_l0_hh.grad += update[hid_dim*2:hid_dim*3, hid_dim:hid_dim*2]
    rnn.str2snr_weight_l0_hh.grad += update[hid_dim*3:hid_dim*4, :hid_dim]
    rnn.stn2snr_weight_l0_hh.grad += update[hid_dim*3:hid_dim*4, hid_dim*2:hid_dim*3]
    rnn.snr2thal_weight_l0_hh.grad += update[hid_dim*4:hid_dim*5, hid_dim*3:hid_dim*4]
    rnn.thal2alm_weight_l0_hh.grad += update[hid_dim*5:hid_dim*6, hid_dim*4:hid_dim*5]
    rnn.alm2alm_weight_l0_hh.grad += update[hid_dim*5:hid_dim*6, hid_dim*5:hid_dim*6]

def simple_dynamics_d1(act, rnn, hid_dim):

    # Get full weights for training
    alm2alm = rnn.alm2alm_weight_l0_hh
    alm2str = rnn.alm2str_weight_l0_hh
    thal2alm = rnn.thal2alm_weight_l0_hh
    thal2str = rnn.thal2str_weight_l0_hh
    str2thal = rnn.str2thal_weight_l0_hh

    # Concatenate into single weight matrix

                        # STR        SNR       Thal      ALM
    W_str = torch.cat([rnn.zeros, thal2str, alm2str], dim=1)          # STR
    W_thal = torch.cat([str2thal, rnn.zeros, rnn.zeros], dim=1)   # Thal
    W_alm = torch.cat([rnn.zeros, thal2alm, alm2alm], dim=1)       # ALM

    # Putting all weights together
    W_rec = torch.cat([W_str, W_thal, W_alm], dim=0)

    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

    update = 1e-3 * W_rec.T * d_act

    rnn.thal2str_weight_l0_hh.grad += update[:hid_dim, hid_dim:hid_dim*2]
    rnn.alm2str_weight_l0_hh.grad += update[:hid_dim, hid_dim*2:hid_dim*3]
    rnn.str2thal_weight_l0_hh.grad += update[hid_dim:hid_dim*2, :hid_dim]
    rnn.thal2alm_weight_l0_hh.grad += update[hid_dim*2:hid_dim*3, hid_dim:hid_dim*2]
    rnn.alm2alm_weight_l0_hh.grad += update[hid_dim*2:hid_dim*3, hid_dim*2:hid_dim*3]

def simple_dynamics_stralm(act, rnn, hid_dim):
    
    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

    rnn.alm2alm_weight_l0_hh.grad += (1e-4 * rnn.alm2alm_weight_l0_hh.T * d_act[hid_dim:])
    rnn.alm2str_weight_l0_hh.grad += (1e-4 * rnn.alm2str_weight_l0_hh.T * d_act[hid_dim:])
    rnn.str2alm_weight_l0_hh.grad += (1e-4 * rnn.str2alm_weight_l0_hh.T * d_act[:hid_dim])