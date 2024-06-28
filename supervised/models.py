import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt

class RNN_MultiRegional_D1D2(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, noise_level=0.01, constrained=True):
        super(RNN_MultiRegional_D1D2, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons in each region
                action_dim: output dimension, should be one for lick or no lick
        '''

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.constrained = constrained

        self.alm_mask = torch.cat([torch.zeros(size=(hid_dim * 5,)), 
                                    torch.ones(size=(hid_dim,))]).cuda()
        self.str_mask = torch.cat([torch.ones(size=(hid_dim,)), 
                                   torch.zeros(size=(hid_dim * 5,))]).cuda()
        self.str_d1_mask = torch.cat([torch.ones(size=(int(hid_dim/2),)), 
                                    torch.zeros(size=(int(hid_dim/2),)),
                                    torch.zeros(size=(hid_dim * 5,))]).cuda()
        self.str_d2_mask = torch.cat([torch.zeros(size=(int(hid_dim/2),)), 
                                    torch.ones(size=(int(hid_dim/2),)),
                                    torch.zeros(size=(hid_dim * 5,))]).cuda()
        self.strthal_mask = torch.cat([torch.zeros(size=(int(hid_dim/4),)),
                                    torch.ones(size=(int(hid_dim/2),)), 
                                    torch.zeros(size=(int(hid_dim/4),)),
                                    torch.zeros(size=(hid_dim * 5,))]).cuda()
        self.tonic_inp = torch.cat([
            torch.zeros(size=(hid_dim,)),
            torch.ones(size=(2*hid_dim,)),
            torch.zeros(size=(hid_dim,)),
            torch.ones(size=(hid_dim,)),
            torch.zeros(size=(hid_dim,))
        ]).cuda()
        
        # Inhibitory Connections
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Mix of Excitatory and Inhibitory Connections
        self.alm2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.alm2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.str2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.str2gpe_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.gpe2stn_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.stn2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.snr2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        if constrained:

            # Initialize weights to be all positive for Dale's Law
            nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.thal2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.str2snr_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.str2gpe_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.gpe2stn_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.stn2snr_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, 0, 0.01)

            # Implement Necessary Masks
            # Striatum recurrent weights
            sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
            nn.init.sparse_(sparse_matrix, 0.9)
            sparse_mask = torch.where(sparse_matrix != 0, 1, 0).cuda()
            self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).cuda()
            self.str2str_fixed = torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.001).cuda() * sparse_mask
            self.str2str_D = -1*torch.eye(hid_dim).cuda()

            self.alm2alm_D = torch.eye(hid_dim).cuda()
            self.alm2alm_D[hid_dim-int(0.3*hid_dim):, 
                            hid_dim-int(0.3*hid_dim):] *= -1
            
            # ALM to striatum weights
            self.alm2str_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim)))
            self.alm2str_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim)))
            self.alm2str_mask = torch.cat([self.alm2str_mask_excitatory, self.alm2str_mask_inhibitory], dim=1).cuda()

            # Thal to STR mask
            self.thal2str_mask = torch.cat([torch.zeros(size=(int(hid_dim/4), hid_dim)),
                                            torch.ones(size=(int(hid_dim/2), hid_dim)),
                                            torch.zeros(size=(int(hid_dim/4), hid_dim))], dim=0).cuda()

            # STR to SNR D
            self.str2snr_D = -1 * torch.eye(hid_dim).cuda()
            self.str2snr_mask = torch.cat([torch.ones(size=(hid_dim, int(hid_dim/2))), 
                                        torch.zeros(size=(hid_dim, int(hid_dim/2)))], dim=1).cuda()

            # SNR to Thal D
            self.snr2thal_D = -1 * torch.eye(hid_dim).cuda()

            # STR to GPE D
            self.str2gpe_D = -1 * torch.eye(hid_dim).cuda()
            self.str2gpe_mask = torch.cat([torch.zeros(size=(hid_dim, int(hid_dim/2))), 
                                        torch.ones(size=(hid_dim, int(hid_dim/2)))], dim=1).cuda()

            # GPE to STN D
            self.gpe2stn_D = -1 * torch.eye(hid_dim).cuda()

        else:

            # Initialize all weights randomly
            nn.init.uniform_(self.str2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.thal2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.str2snr_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.str2gpe_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.gpe2stn_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.stn2snr_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, -0.01, 0.01)

        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim * 6)))
        nn.init.uniform_(self.inp_weight, 0, 0.1)

        # Zeros for no weights
        self.zeros = torch.zeros(size=(hid_dim, hid_dim)).cuda()

        # Time constants for networks (not sure what would be biologically plausible?)
        self.t_const = 0.01

        # Noise level
        self.sigma_recur = noise_level
        self.sigma_input = noise_level

    def forward(self, inp, hn, x, inhib_stim, noise=True):

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

        if self.constrained:

            # Get full weights for training
            str2str = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, 1e-15, 1) + self.str2str_fixed) @ self.str2str_D
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-15, 1) @ self.alm2alm_D
            alm2str = self.alm2str_mask * F.hardtanh(self.alm2str_weight_l0_hh, 1e-15, 1)
            thal2alm = F.hardtanh(self.thal2alm_weight_l0_hh, 1e-15, 1)
            thal2str = self.thal2str_mask * F.hardtanh(self.thal2str_weight_l0_hh, 1e-15, 1)
            str2snr = (self.str2snr_mask * F.hardtanh(self.str2snr_weight_l0_hh, 1e-15, 1)) @ self.str2snr_D
            str2gpe = (self.str2gpe_mask * F.hardtanh(self.str2gpe_weight_l0_hh, 1e-15, 1)) @ self.str2gpe_D
            gpe2stn = F.hardtanh(self.gpe2stn_weight_l0_hh, 1e-15, 1) @ self.gpe2stn_D
            stn2snr = F.hardtanh(self.stn2snr_weight_l0_hh, 1e-15, 1)
            snr2thal = F.hardtanh(self.snr2thal_weight_l0_hh, 1e-15, 1) @ self.snr2thal_D
            inp_weight = F.hardtanh(self.inp_weight, 1e-15, 1)

            # Concatenate into single weight matrix

                                # STR       GPE         STN         SNR       Thal      ALM
            W_str = torch.cat([str2str, self.zeros, self.zeros, self.zeros, thal2str, alm2str], dim=1)          # STR
            W_gpe = torch.cat([str2gpe, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros], dim=1)     # GPE
            W_stn = torch.cat([self.zeros, gpe2stn, self.zeros, self.zeros, self.zeros, self.zeros], dim=1)     # STN
            W_snr = torch.cat([str2snr, self.zeros, stn2snr, self.zeros, self.zeros, self.zeros], dim=1)        # SNR
            W_thal = torch.cat([self.zeros, self.zeros, self.zeros, snr2thal, self.zeros, self.zeros], dim=1)   # Thal
            W_alm = torch.cat([self.zeros, self.zeros, self.zeros, self.zeros, thal2alm, alm2alm], dim=1)       # ALM

        else:

            # Concatenate into single weight matrix

                                # STR       GPE         STN         SNR       Thal      ALM
            W_str = torch.cat([self.str2str_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.thal2str_weight_l0_hh, self.alm2str_weight_l0_hh], dim=1)            # STR
            W_gpe = torch.cat([self.str2gpe_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros], dim=1)                                           # GPE
            W_stn = torch.cat([self.zeros, self.gpe2stn_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.zeros], dim=1)                                           # STN
            W_snr = torch.cat([self.str2snr_weight_l0_hh, self.zeros, self.stn2snr_weight_l0_hh, self.zeros, self.zeros, self.zeros], dim=1)                            # SNR
            W_thal = torch.cat([self.zeros, self.zeros, self.zeros, self.snr2thal_weight_l0_hh, self.zeros, self.zeros], dim=1)                                         # Thal
            W_alm = torch.cat([self.zeros, self.zeros, self.zeros, self.zeros, self.thal2alm_weight_l0_hh, self.alm2alm_weight_l0_hh], dim=1)                           # ALM

        # Putting all weights together
        W_rec = torch.cat([W_str, W_gpe, W_stn, W_snr, W_thal, W_alm], dim=0)

        # Loop through RNN
        for t in range(size):

            if noise:
                perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * np.random.normal(0, 1)
                perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * np.random.normal(0, 1)
            else:
                perturb_hid = 0
                perturb_inp = 0

            if self.constrained:

                hn_next = F.relu(hn_next 
                        + self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ inp_weight * self.strthal_mask) + inhib_stim[:, t, :] + self.tonic_inp) 
                        + perturb_hid)
            
            else:
            
                hn_next = F.relu(hn_next 
                        + self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ inp_weight * self.strthal_mask) + inhib_stim[:, t, :]) 
                        + perturb_hid)

            new_hs.append(hn_next)
            new_xs.append(x_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        x_out = torch.stack(new_xs, dim=1)

        hn_last = rnn_out[:, -1, :].unsqueeze(0)
        x_last = x_out[:, -1, :].unsqueeze(0)

        return hn_last, rnn_out, x_last, x_out


class RNN_MultiRegional_STRALM(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, noise_level=0.01, constrained=True):
        super(RNN_MultiRegional_STRALM, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons in each region
                action_dim: output dimension, should be one for lick or no lick
        '''

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.constrained = constrained

        self.alm_mask = torch.cat([torch.zeros(size=(hid_dim,)), 
                                    torch.ones(size=(hid_dim,))]).cuda()
        self.str_d1_mask = torch.cat([torch.ones(size=(hid_dim,)), 
                                   torch.zeros(size=(hid_dim,))]).cuda()
        
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.alm2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.alm2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.str2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim * 2)))

        if constrained:

            nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.str2alm_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.inp_weight, 0, 0.01)
        
            # Implement Necessary Masks
            # Striatum recurrent weights
            sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
            nn.init.sparse_(sparse_matrix, 0.9)
            sparse_mask = torch.where(sparse_matrix != 0, 1, 0).cuda()
            self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).cuda()
            self.str2str_fixed = torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.001).cuda() * sparse_mask
            self.str2str_D = -1*torch.eye(hid_dim).cuda()

            self.alm2alm_D = torch.eye(hid_dim).cuda()
            self.alm2alm_D[hid_dim-int(0.3*hid_dim):, 
                            hid_dim-int(0.3*hid_dim):] *= -1
            
            # ALM to striatum weights
            self.alm2str_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim)))
            self.alm2str_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim)))
            self.alm2str_mask = torch.cat([self.alm2str_mask_excitatory, self.alm2str_mask_inhibitory], dim=1).cuda()

        else:

            nn.init.uniform_(self.str2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.str2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.inp_weight, -0.01, 0.01)

        # Zeros for no weights
        self.zeros = torch.zeros(size=(hid_dim, hid_dim)).cuda()

        # Time constants for networks (not sure what would be biologically plausible?)
        self.t_const = 0.01

        # Noise level
        self.sigma_recur = noise_level
        self.sigma_input = noise_level

    def forward(self, inp, hn, x, inhib_stim, noise=True):

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

        if self.constrained:

            str2str = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, 1e-15, 1) + self.str2str_fixed) @ self.str2str_D
            str2alm = F.hardtanh(self.str2alm_weight_l0_hh, 1e-15, 1)
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-15, 1) @ self.alm2alm_D
            alm2str = self.alm2str_mask * F.hardtanh(self.alm2str_weight_l0_hh, 1e-15, 1)

            # Concatenate into single weight matrix
            W_str = torch.cat([str2str, alm2str], dim=1)
            W_alm = torch.cat([str2alm, alm2alm], dim=1)
            W_rec = torch.cat([W_str, W_alm], dim=0)
        
        else:
            
            # Concatenate into single weight matrix
            W_str = torch.cat([self.str2str_weight_l0_hh, self.alm2str_weight_l0_hh], dim=1)
            W_alm = torch.cat([self.str2alm_weight_l0_hh, self.alm2alm_weight_l0_hh], dim=1)
            W_rec = torch.cat([W_str, W_alm], dim=0)

        # Loop through RNN
        for t in range(size):

            if noise:
                perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * np.random.normal(0, 1)
                perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * np.random.normal(0, 1)
            else:
                perturb_hid = 0
                perturb_inp = 0

            hn_next = F.relu(hn_next + 
                      self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ self.inp_weight * self.str_d1_mask) + inhib_stim[:, t, :]) 
                      + perturb_hid)

            new_hs.append(hn_next)
            new_xs.append(x_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        x_out = torch.stack(new_xs, dim=1)

        hn_last = rnn_out[:, -1, :].unsqueeze(0)
        x_last = x_out[:, -1, :].unsqueeze(0)

        return hn_last, rnn_out, x_last, x_out


class RNN_MultiRegional_D1(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, noise_level=0.01, constrained=True):
        super(RNN_MultiRegional_D1, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons in each region
                action_dim: output dimension, should be one for lick or no lick
        '''

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.constrained = constrained

        self.alm_mask = torch.cat([torch.zeros(size=(hid_dim * 3,)), 
                                    torch.ones(size=(hid_dim,))]).cuda()
        self.str_d1_mask = torch.cat([torch.ones(size=(hid_dim,)), 
                                   torch.zeros(size=(hid_dim * 3,))]).cuda()
        self.strthal_mask = torch.cat([torch.zeros(size=(int(hid_dim/2),)),
                                    torch.ones(size=(int(hid_dim/2),)), 
                                    torch.zeros(size=(hid_dim * 3,))]).cuda()
        self.tonic_inp = torch.cat([
            torch.zeros(size=(hid_dim,)),
            0.45 * torch.ones(size=(hid_dim,)),
            torch.ones(size=(hid_dim,)),
            torch.zeros(size=(hid_dim,))
        ]).cuda()
        
        # Inhibitory Connections
        self.str2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Mix of Excitatory and Inhibitory Connections
        self.alm2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.alm2str_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.str2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.snr2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        if constrained:

            nn.init.uniform_(self.str2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.thal2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.str2snr_weight_l0_hh, 0, 0.01)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, 0, 0.01)

            # Implement Necessary Masks
            # Striatum recurrent weights
            sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
            nn.init.sparse_(sparse_matrix, 0.9)
            sparse_mask = torch.where(sparse_matrix != 0, 1, 0).cuda()
            self.str2str_mask = torch.zeros_like(self.str2str_weight_l0_hh).cuda()
            self.str2str_fixed = torch.empty_like(self.str2str_weight_l0_hh).uniform_(0, 0.001).cuda() * sparse_mask
            self.str2str_D = -1*torch.eye(hid_dim).cuda()

            self.alm2alm_D = torch.eye(hid_dim).cuda()
            self.alm2alm_D[hid_dim-int(0.3*hid_dim):, 
                            hid_dim-int(0.3*hid_dim):] *= -1
            
            # ALM to striatum weights
            self.alm2str_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim)))
            self.alm2str_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim)))
            self.alm2str_mask = torch.cat([self.alm2str_mask_excitatory, self.alm2str_mask_inhibitory], dim=1).cuda()

            # Thal to STR mask
            self.thal2str_mask = torch.cat([torch.zeros(size=(int(hid_dim/2), hid_dim)),
                                            torch.ones(size=(int(hid_dim/2), hid_dim))], dim=0).cuda()

            # STR to SNR D
            self.str2snr_D = -1 * torch.eye(hid_dim).cuda()

            # SNR to Thal D
            self.snr2thal_D = -1 * torch.eye(hid_dim).cuda()

        else:

            nn.init.uniform_(self.str2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.thal2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.str2snr_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, -0.01, 0.01)

        # Input weights
        self.inp_weight = nn.Parameter(torch.empty(size=(inp_dim, hid_dim * 4)))
        nn.init.uniform_(self.inp_weight, 0, 0.1)

        # Zeros for no weights
        self.zeros = torch.zeros(size=(hid_dim, hid_dim)).cuda()

        # Time constants for networks (not sure what would be biologically plausible?)
        self.t_const = 0.01

        # Noise level
        self.sigma_recur = noise_level
        self.sigma_input = noise_level

    def forward(self, inp, hn, x, inhib_stim, noise=True):

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

        if self.constrained:

            str2str = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, 1e-15, 1) + self.str2str_fixed) @ self.str2str_D
            str2snr = F.hardtanh(self.str2snr_weight_l0_hh, 1e-15, 1) @ self.str2snr_D
            snr2thal = F.hardtanh(self.snr2thal_weight_l0_hh, 1e-15, 1) @ self.snr2thal_D
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-15, 1) @ self.alm2alm_D
            alm2str = self.alm2str_mask * F.hardtanh(self.alm2str_weight_l0_hh, 1e-15, 1)
            thal2str = self.thal2str_mask * F.hardtanh(self.thal2str_weight_l0_hh, 1e-15, 1)
            thal2alm = F.hardtanh(self.thal2alm_weight_l0_hh, 1e-15, 1)

            # Concatenate into single weight matrix
            W_str = torch.cat([str2str, self.zeros, thal2str, alm2str], dim=1)
            W_snr = torch.cat([str2snr, self.zeros, self.zeros, self.zeros], dim=1)
            W_thal = torch.cat([self.zeros, snr2thal, self.zeros, self.zeros], dim=1)
            W_alm = torch.cat([self.zeros, self.zeros, thal2alm, alm2alm], dim=1)
            W_rec = torch.cat([W_str, W_snr, W_thal, W_alm], dim=0)
        
        else:

            # Concatenate into single weight matrix
            W_str = torch.cat([self.str2str_weight_l0_hh, self.zeros, self.thal2str_weight_l0_hh, self.alm2str_weight_l0_hh], dim=1)
            W_snr = torch.cat([self.str2snr_weight_l0_hh, self.zeros, self.zeros, self.zeros], dim=1)
            W_thal = torch.cat([self.zeros, self.snr2thal_weight_l0_hh, self.zeros, self.zeros], dim=1)
            W_alm = torch.cat([self.zeros, self.zeros, self.thal2alm_weight_l0_hh, self.alm2alm_weight_l0_hh], dim=1)
            W_rec = torch.cat([W_str, W_snr, W_thal, W_alm], dim=0)

        # Loop through RNN
        for t in range(size):

            if noise:
                perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * np.random.normal(0, 1)
                perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * np.random.normal(0, 1)
            else:
                perturb_hid = 0
                perturb_inp = 0

            if self.constrained:

                hn_next = F.relu(hn_next + 
                        self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ self.inp_weight * self.strthal_mask) + inhib_stim[:, t, :] + self.tonic_inp) 
                        + perturb_hid)
            
            else:

                hn_next = F.relu(hn_next + 
                        self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ self.inp_weight * self.strthal_mask) + inhib_stim[:, t, :]) 
                        + perturb_hid)

            new_hs.append(hn_next)
            new_xs.append(x_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        x_out = torch.stack(new_xs, dim=1)

        hn_last = rnn_out[:, -1, :].unsqueeze(0)
        x_last = x_out[:, -1, :].unsqueeze(0)

        return hn_last, rnn_out, x_last, x_out