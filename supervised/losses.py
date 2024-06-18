import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import RNN_MultiRegional, RNN_MultiRegional_NoConstraint
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_delay_data, get_ramp

def loss_constraint(criterion, constraint_criterion, act, out, neural_act, y_data, hid_dim, alm_start, str_start, thal_start):
    
    loss = (1e-2 * criterion(out, y_data) 
            + constraint_criterion(torch.mean(act[:, :, alm_start:alm_start+hid_dim], dim=-1, keepdim=True), neural_act)
            + 1e-4 * torch.mean(torch.pow(act, 2), dim=(1, 2, 0))  
            + constraint_criterion(torch.mean(act[:, :, str_start:str_start+hid_dim], dim=-1, keepdim=True), neural_act)
            + constraint_criterion(torch.mean(act[:, :, thal_start:thal_start+hid_dim], dim=-1, keepdim=True), neural_act)
            )

    return loss

def loss_no_constraint(constraint_criterion, act, neural_act, alm_start, str_start):
    
    loss = (#constraint_criterion(out, 5*neural_act) 
            + constraint_criterion(torch.mean(act[:, :, alm_start:], dim=-1, keepdim=True), neural_act)
            + 1e-4 * torch.mean(torch.pow(act, 2), dim=(1, 2, 0))  
            + constraint_criterion(torch.mean(act[:, :, str_start:alm_start], dim=-1, keepdim=True), neural_act)
            )

    return loss

def simple_dynamics_constraint(act, rnn, hid_dim):

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

    W_str_grad = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, rnn.thal2str_weight_l0_hh.grad, rnn.alm2str_weight_l0_hh.grad], dim=1)          # STR
    W_gpe_grad = torch.cat([rnn.str2gpe_weight_l0_hh.grad, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # GPE
    W_stn_grad = torch.cat([rnn.zeros, rnn.gpe2stn_weight_l0_hh.grad, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)     # STN
    W_snr_grad = torch.cat([rnn.str2snr_weight_l0_hh.grad, rnn.zeros, rnn.stn2snr_weight_l0_hh.grad, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)        # SNR
    W_thal_grad = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, rnn.snr2thal_weight_l0_hh.grad, rnn.zeros, rnn.zeros], dim=1)   # Thal
    W_alm_grad = torch.cat([rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros, rnn.thal2alm_weight_l0_hh.grad, rnn.alm2alm_weight_l0_hh.grad], dim=1)       # ALM

    # Putting all weights together
    W_rec_grad = torch.cat([W_str_grad, W_gpe_grad, W_stn_grad, W_snr_grad, W_thal_grad, W_alm_grad], dim=0)
    
    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

    W_rec_grad += (1e-4 * W_rec.T * d_act)

def simple_dynamics_no_constraint(act, rnn, alm_start, str_start):
    
    # Penalize complex trajectories
    d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

    rnn.alm2alm_weight_l0_hh.grad += (1e-4 * rnn.alm2alm_weight_l0_hh.T * d_act[alm_start:])
    rnn.alm2str_weight_l0_hh.grad += (1e-4 * rnn.alm2str_weight_l0_hh.T * d_act[alm_start:])
    rnn.str2alm_weight_l0_hh.grad += (1e-4 * rnn.str2alm_weight_l0_hh.T * d_act[str_start:alm_start])
    rnn.str2str_weight_l0_hh.grad += (1e-4 * rnn.str2str_weight_l0_hh.T * d_act[str_start:alm_start])