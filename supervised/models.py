import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

class RNN_MultiRegional(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim):
        super(RNN_MultiRegional, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons, each region and connection between region has hid_dim/2 neurons
                action_dim: output dimension, should be one for lick or no lick
        '''

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.alm_mask = torch.cat([torch.zeros(size=(int(hid_dim * (3/4)),)), 
                                    torch.ones(size=(int(hid_dim/4),))]).cuda()
        self.str_mask = torch.cat([torch.ones(size=(int(hid_dim/4),)), 
                                   torch.zeros(size=(int(hid_dim * (3/4)),))]).cuda()
        self.thal_mask = torch.cat([torch.zeros(size=(int(hid_dim/2),)),
                                    torch.ones(size=(int(hid_dim/4),)),
                                    torch.zeros(size=(int(hid_dim/4),))]).cuda()
        self.tonic_inp = torch.cat([torch.zeros([int(hid_dim/4),]), torch.ones([int(hid_dim*(1/2)),]), torch.zeros([int(hid_dim/4),])]).cuda()
        
        # Inhibitory Connections
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Excitatory Connections
        self.thal2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Excitatory Connections
        self.thal2str_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Mix of Excitatory and Inhibitory Connections
        self.alm2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Excitatory Connections
        self.alm2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Excitatory Connections
        self.alm2str_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Inhibitory Connections
        self.str2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Inhibitory Connections
        self.snr2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))
        # Mix
        self.thalgating_weights = nn.Parameter(torch.empty(size=(int(hid_dim/4), int(hid_dim/4))))

        nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.thal2alm_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.thal2str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.alm2alm_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.alm2str_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.alm2thal_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.str2snr_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.snr2thal_weight_l0_hh, 0, 0.01)
        nn.init.uniform_(self.thalgating_weights, -0.01, 0.01)

        # Implement Necessary Masks
        # Striatum recurrent weights
        sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
        nn.init.sparse_(sparse_matrix, 0.85)
        sparse_mask = torch.where(sparse_matrix != 0, 1, 0).cuda()
        self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).cuda()
        self.str2str_fixed = torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.001).cuda() * sparse_mask
        self.str2str_D = -1*torch.eye(int(hid_dim/4)).cuda()

        self.alm2alm_D = torch.eye(int(hid_dim/4)).cuda()
        self.alm2alm_D[int(hid_dim/4)-(int( 0.3*(hid_dim/4) )):, 
                        int(hid_dim/4)-(int( 0.3*(hid_dim/4) )):] *= -1
        
        # ALM to striatum weights
        self.alm2str_mask_excitatory = torch.ones(size=(int(hid_dim/4), int(hid_dim/4) - int(0.8*(hid_dim/4))))
        self.alm2str_mask_inhibitory = torch.zeros(size=(int(hid_dim/4), int(0.8*(hid_dim/4))))
        self.alm2str_mask = torch.cat([self.alm2str_mask_excitatory, self.alm2str_mask_inhibitory], dim=1).cuda()

        # Striatum to SNr weights
        self.str2snr_D = -1*torch.eye(int(hid_dim/4)).cuda()

        # SNr to Thalamus weights
        self.snr2thal_D = -1*torch.eye(int(hid_dim/4)).cuda()
        
        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        nn.init.uniform_(self.inp_weight, 0, 0.1)

        # Zeros for no weights
        self.zeros = torch.zeros(size=(int(self.hid_dim/4), int(self.hid_dim/4))).cuda()

        # Behavioral output layer
        self.fc1 = nn.Linear(hid_dim, action_dim)

        # Time constants for networks (not sure what would be biologically plausible?)
        self.t_const = 0.1*torch.ones(size=(hid_dim,)).cuda()

    def forward(self, inp, hn, x):

        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''

        # Saving hidden states
        hn_next = hn.squeeze(0)
        x_next = x.squeeze(0)
        size = inp.shape[1]
        new_hs = []
        new_xs = []

        # Get full weights for training
        str2str = (self.str2str_mask * F.relu(self.str2str_weight_l0_hh) + self.str2str_fixed) @ self.str2str_D
        alm2alm = F.relu(self.alm2alm_weight_l0_hh) @ self.alm2alm_D
        alm2str = self.alm2str_mask * F.relu(self.alm2str_weight_l0_hh)
        thal2alm = F.relu(self.thal2alm_weight_l0_hh)
        thal2str = F.relu(self.thal2str_weight_l0_hh)
        str2snr = F.relu(self.str2snr_weight_l0_hh) @ self.str2snr_D
        snr2thal = F.relu(self.snr2thal_weight_l0_hh) #@ self.snr2thal_D

        # Concatenate into single weight matrix
        W_str = torch.cat([str2str, self.zeros, thal2str, alm2str], dim=1)
        W_snr = torch.cat([self.zeros, self.zeros, self.zeros, self.zeros], dim=1)
        W_thal = torch.cat([snr2thal, self.zeros, self.zeros, self.zeros], dim=1)
        W_alm = torch.cat([self.zeros, self.zeros, thal2alm, alm2alm], dim=1)
        W_rec = torch.cat([W_str, W_snr, W_thal, W_alm], dim=0)

        # Loop through RNN
        for t in range(size):
            gate = torch.cat([F.sigmoid(self.thalgating_weights @ x_next[:, int(self.hid_dim/2):int(self.hid_dim*(3/4))].T).T, 
                                torch.ones(size=(hn_next.shape[0], int(self.hid_dim*(3/4))), device="cuda")], dim=1)
            x_next = x_next + self.t_const * gate * (-x_next + (W_rec @ hn_next.T).T + (inp[:, t, :] @ self.inp_weight * self.str_mask)) 
            hn_next = F.relu(x_next) 
            new_hs.append(hn_next)
            new_xs.append(x_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        x_out = torch.stack(new_xs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)
        x_last = x_out[:, -1, :].unsqueeze(0)

        # Behavioral output layer
        out = torch.sigmoid(self.fc1(rnn_out * self.alm_mask))
        
        return out, hn_last, rnn_out, x_last, x_out