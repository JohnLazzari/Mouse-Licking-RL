import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt

class RNN_MultiRegional_D1D2(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, noise_level_act=0.01, noise_level_inp=0.01, constrained=True):
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
        self.fsi_size = int(hid_dim * 0.3)

        self.alm_ramp_mask = torch.cat([torch.zeros(size=(hid_dim * 5,)), 
                                    torch.ones(size=(hid_dim - int(hid_dim * 0.3),)),
                                    torch.zeros(size=(int(hid_dim * 0.3),)),
                                    torch.zeros(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.alm_inhib_mask = torch.cat([torch.zeros(size=(hid_dim * 5,)), 
                                    torch.zeros(size=(hid_dim - int(hid_dim * 0.3),)),
                                    torch.ones(size=(int(hid_dim * 0.3),)),
                                    torch.zeros(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.full_alm_mask = torch.cat([torch.zeros(size=(hid_dim * 5,)), 
                                    torch.ones(size=(hid_dim,)),
                                    torch.ones(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.iti_mask = torch.cat([torch.zeros(size=(hid_dim * 6,)), 
                                    torch.ones(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.str_mask = torch.cat([torch.ones(size=(hid_dim,)), 
                                   torch.zeros(size=(hid_dim * 5,)),
                                    torch.zeros(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.str_d1_mask = torch.cat([torch.ones(size=(int(hid_dim/2),)), 
                                    torch.zeros(size=(int(hid_dim/2),)),
                                    torch.zeros(size=(hid_dim * 5,)),
                                    torch.zeros(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.str_d2_mask = torch.cat([torch.zeros(size=(int(hid_dim/2),)), 
                                    torch.ones(size=(int(hid_dim/2),)),
                                    torch.zeros(size=(hid_dim * 5,)),
                                    torch.zeros(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.thal_mask = torch.cat([torch.zeros(size=(hid_dim * 4,)),
                                    torch.ones(size=(hid_dim,)),
                                    torch.zeros(size=(hid_dim,)),
                                    torch.zeros(size=(inp_dim,)),
                                    torch.zeros(size=(self.fsi_size,)),
                                    ]).cuda()

        self.tonic_inp_str = torch.zeros(size=(hid_dim,), device="cuda")
        self.tonic_inp_gpe = torch.ones(size=(hid_dim,), device="cuda")
        self.tonic_inp_stn = torch.ones(size=(hid_dim,), device="cuda")
        self.tonic_inp_snr = torch.zeros(size=(hid_dim,), device="cuda")
        self.tonic_inp_thal_int = torch.ones(size=(int(hid_dim/2),), device="cuda")
        self.tonic_inp_thal_alm = torch.ones(size=(int(hid_dim/2),), device="cuda")
        self.tonic_inp_alm = torch.zeros(size=(hid_dim,), device="cuda")
        self.tonic_inp_iti = torch.zeros(size=(inp_dim,), device="cuda")
        self.tonic_inp_fsi = torch.zeros(size=(self.fsi_size,), device="cuda")

        self.tonic_inp = torch.cat([
            self.tonic_inp_str,
            self.tonic_inp_gpe,
            self.tonic_inp_stn,
            self.tonic_inp_snr,
            self.tonic_inp_thal_int,
            self.tonic_inp_thal_alm,
            self.tonic_inp_alm,
            self.tonic_inp_iti,
            self.tonic_inp_fsi
        ])
        
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
        # Excitatory Connections
        self.alm2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
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
        # Inhibitory Connections
        self.fsi2str_weight = nn.Parameter(torch.empty(size=(hid_dim, self.fsi_size)))
        # Excitatory Connections
        self.thal2fsi_weight = nn.Parameter(torch.empty(size=(self.fsi_size, hid_dim)))
        # Excitatory Connections
        self.alm2fsi_weight = nn.Parameter(torch.empty(size=(self.fsi_size, hid_dim)))
        # Excitatory Connections
        self.iti2fsi_weight = nn.Parameter(torch.empty(size=(self.fsi_size, inp_dim)))
        # Excitatory Connections
        self.fsi2fsi_weight = nn.Parameter(torch.empty(size=(self.fsi_size, self.fsi_size)))

        if constrained:

            # Initialize weights to be all positive for Dale's Law
            nn.init.uniform_(self.str2str_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.thal2str_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.alm2str_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.alm2thal_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.str2snr_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.str2gpe_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.gpe2stn_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.stn2snr_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.fsi2str_weight, 0, 1e-2)
            nn.init.uniform_(self.thal2fsi_weight, 0, 1e-2)
            nn.init.uniform_(self.alm2fsi_weight, 0, 1e-2)
            nn.init.uniform_(self.iti2fsi_weight, 0, 1e-2)
            nn.init.uniform_(self.fsi2fsi_weight, 0, 1e-2)

            # Implement Necessary Masks
            # Striatum recurrent weights
            sparse_matrix = torch.empty_like(self.str2str_weight_l0_hh)
            nn.init.sparse_(sparse_matrix, 0.9)
            self.str2str_sparse_mask = torch.where(sparse_matrix != 0, 1, 0).cuda()
            self.str2str_D = -1*torch.eye(hid_dim).cuda()

            d1_lateral_connections_mask = torch.cat([
                torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
                torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
            ], dim=1)

            d2_lateral_connections_mask = torch.cat([
                torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
                torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
            ], dim=1)

            self.str2str_split_mask = torch.cat([
                d1_lateral_connections_mask,
                d2_lateral_connections_mask
            ], dim=0).cuda()

            self.alm2alm_D = torch.eye(hid_dim).cuda()
            self.alm2alm_D[hid_dim-int(0.3*hid_dim):, 
                            hid_dim-int(0.3*hid_dim):] *= -1
            
            # ALM to striatum weights
            self.alm2str_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim)))
            self.alm2str_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim)))
            
            self.alm2str_mask = torch.cat([
                self.alm2str_mask_excitatory, 
                self.alm2str_mask_inhibitory
            ], dim=1).cuda()

            # ALM to Thal mask
            self.alm2thal_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim)))
            self.alm2thal_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim)))
            
            self.alm2thal_mask = torch.cat([
                self.alm2thal_mask_excitatory, 
                self.alm2thal_mask_inhibitory
            ], dim=1).cuda()
            
            # STR to SNR D
            self.str2snr_D = -1 * torch.eye(hid_dim).cuda()
            
            self.str2snr_mask = torch.cat([
                torch.ones(size=(hid_dim, int(hid_dim/2))), 
                torch.zeros(size=(hid_dim, int(hid_dim/2)))
            ], dim=1).cuda()

            # SNR to Thal D
            self.snr2thal_D = -1 * torch.eye(hid_dim).cuda()

            # STR to GPE D
            self.str2gpe_D = -1 * torch.eye(hid_dim).cuda()
            self.str2gpe_mask = torch.cat([torch.zeros(size=(hid_dim, int(hid_dim/2))), 
                                        torch.ones(size=(hid_dim, int(hid_dim/2)))], dim=1).cuda()

            # GPE to STN D
            self.gpe2stn_D = -1 * torch.eye(hid_dim).cuda()

            # FSI to STR D
            self.fsi2str_D = -1 * torch.eye(self.fsi_size).cuda()

            # FSI to FSI D
            self.fsi2fsi_D = -1 * torch.eye(self.fsi_size).cuda()
            
        else:

            # Initialize all weights randomly
            nn.init.uniform_(self.str2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.thal2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2str_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.alm2thal_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.str2snr_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.str2gpe_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.gpe2stn_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.stn2snr_weight_l0_hh, -0.01, 0.01)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, -0.01, 0.01)

        # Input weights STR
        self.inp_weight_str = nn.Parameter(torch.empty(size=(hid_dim, inp_dim)))
        nn.init.uniform_(self.inp_weight_str, 0, 1e-2)
        
        # Zeros for no weights
        self.zeros = torch.zeros(size=(hid_dim, hid_dim), device="cuda")
        self.zeros_to_iti = torch.zeros(size=(inp_dim, hid_dim), device="cuda")
        self.zeros_from_iti = torch.zeros(size=(hid_dim, inp_dim), device="cuda")
        self.zeros_rec_iti = torch.zeros(size=(inp_dim, inp_dim), device="cuda")

        self.zeros_to_fsi = torch.zeros(size=(self.fsi_size, hid_dim), device="cuda")
        self.zeros_from_fsi = torch.zeros(size=(hid_dim, self.fsi_size), device="cuda")
        self.zeros_from_fsi2iti = torch.zeros(size=(inp_dim, self.fsi_size), device="cuda")

        # Time constants for networks
        self.t_const = 0.01

        # Noise level
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp

    def forward(self, inp, cue_inp, hn, xn, inhib_stim, noise=True):

        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''

        # Saving hidden states
        hn_next = hn.squeeze(0)
        xn_next = xn.squeeze(0)
        size = inp.shape[1]
        new_hs = []
        new_xs = []

        if self.constrained:

            # Get full weights for training
            str2str = (self.str2str_sparse_mask * F.hardtanh(self.str2str_weight_l0_hh, 1e-10, 1)) @ self.str2str_D
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-10, 1) @ self.alm2alm_D
            alm2str = self.alm2str_mask * F.hardtanh(self.alm2str_weight_l0_hh, 1e-10, 1)
            alm2thal = self.alm2thal_mask * F.hardtanh(self.alm2thal_weight_l0_hh, 1e-10, 1)
            thal2alm = F.hardtanh(self.thal2alm_weight_l0_hh, 1e-10, 1)
            thal2str = F.hardtanh(self.thal2str_weight_l0_hh, 1e-10, 1)
            str2snr = (self.str2snr_mask * F.hardtanh(self.str2snr_weight_l0_hh, 1e-10, 1)) @ self.str2snr_D
            str2gpe = (self.str2gpe_mask * F.hardtanh(self.str2gpe_weight_l0_hh, 1e-10, 1)) @ self.str2gpe_D
            gpe2stn = F.hardtanh(self.gpe2stn_weight_l0_hh, 1e-10, 1) @ self.gpe2stn_D
            stn2snr = F.hardtanh(self.stn2snr_weight_l0_hh, 1e-10, 1)
            snr2thal = F.hardtanh(self.snr2thal_weight_l0_hh, 1e-10, 1) @ self.snr2thal_D
            fsi2str = F.hardtanh(self.fsi2str_weight, 1e-10, 1) @ self.fsi2str_D
            thal2fsi = F.hardtanh(self.thal2fsi_weight, 1e-10, 1)
            alm2fsi = F.hardtanh(self.alm2fsi_weight, 1e-10, 1)
            iti2fsi = F.hardtanh(self.iti2fsi_weight, 1e-10, 1)
            fsi2fsi = F.hardtanh(self.fsi2fsi_weight, 1e-10, 1) @ self.fsi2fsi_D
            inp_weight_str = F.hardtanh(self.inp_weight_str, 1e-10, 1)

            # Concatenate into single weight matrix

                                # STR       GPE         STN         SNR       Thal      ALM         ALM ITI
            W_str = torch.cat([str2str, self.zeros, self.zeros, self.zeros, thal2str, alm2str, inp_weight_str, fsi2str], dim=1)          # STR
            W_gpe = torch.cat([str2gpe, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros_from_iti, self.zeros_from_fsi], dim=1)     # GPE
            W_stn = torch.cat([self.zeros, gpe2stn, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros_from_iti, self.zeros_from_fsi], dim=1)     # STN
            W_snr = torch.cat([str2snr, self.zeros, stn2snr, self.zeros, self.zeros, self.zeros, self.zeros_from_iti, self.zeros_from_fsi], dim=1)        # SNR
            W_thal = torch.cat([self.zeros, self.zeros, self.zeros, snr2thal, self.zeros, self.zeros, self.zeros_from_iti, self.zeros_from_fsi], dim=1)   # Thal
            W_alm = torch.cat([self.zeros, self.zeros, self.zeros, self.zeros, thal2alm, alm2alm, self.zeros_from_iti, self.zeros_from_fsi], dim=1)       # ALM
            W_alm_iti = torch.cat([self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_rec_iti, self.zeros_from_fsi2iti], dim=1)       # ALM
            W_fsi = torch.cat([self.zeros_to_fsi, self.zeros_to_fsi, self.zeros_to_fsi, self.zeros_to_fsi, thal2fsi, alm2fsi, iti2fsi, fsi2fsi], dim=1)       # ALM

        else:

            # Concatenate into single weight matrix

                                # STR       GPE         STN         SNR       Thal      ALM
            W_str = torch.cat([self.str2str_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.thal2str_weight_l0_hh, self.alm2str_weight_l0_hh, self.inp_weight], dim=1)            # STR
            W_gpe = torch.cat([self.str2gpe_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros_inp_from], dim=1)                                           # GPE
            W_stn = torch.cat([self.zeros, self.gpe2stn_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros_inp_from], dim=1)                                           # STN
            W_snr = torch.cat([self.str2snr_weight_l0_hh, self.zeros, self.stn2snr_weight_l0_hh, self.zeros, self.zeros, self.zeros, self.zeros_inp_from], dim=1)                            # SNR
            W_thal = torch.cat([self.zeros, self.zeros, self.zeros, self.snr2thal_weight_l0_hh, self.zeros, self.alm2thal_weight_l0_hh, self.zeros_inp_from], dim=1)                                         # Thal
            W_alm = torch.cat([self.zeros, self.zeros, self.zeros, self.zeros, self.thal2alm_weight_l0_hh, self.alm2alm_weight_l0_hh, self.zeros_inp_from], dim=1)                           # ALM
            W_alm_iti = torch.cat([self.zeros_inp_to, self.zeros_inp_to, self.zeros_inp_to, self.zeros_inp_to, self.zeros_inp_to, self.zeros_inp_to, self.zeros_inp_rec], dim=1)       # ALM ITI

        # Putting all weights together
        W_rec = torch.cat([W_str, W_gpe, W_stn, W_snr, W_thal, W_alm, W_alm_iti, W_fsi], dim=0)

        # Loop through RNN
        for t in range(size):

            # Add noise to the system if specified
            if noise and t > 1000:
                perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * np.random.normal(0, 1)
                perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * np.random.normal(0, 1)
            else:
                perturb_hid = 0
                perturb_inp = 0

            # Get the ITI mode input to the network
            iti_act = inp[:, t, :] + perturb_inp
            non_iti_mask = torch.zeros(size=(iti_act.shape[0], self.hid_dim * 6), device="cuda")
            non_iti_mask_fsi = torch.zeros(size=(iti_act.shape[0], self.fsi_size), device="cuda")
            iti_input = torch.cat([non_iti_mask, iti_act, non_iti_mask_fsi], dim=-1)

            # Get the activity of the next hidden state
            if self.constrained:

                xn_next = (xn_next 
                            + self.t_const * (
                            -xn_next
                            + (W_rec @ hn_next.T).T
                            + iti_input
                            + self.tonic_inp
                            + inhib_stim[:, t, :]
                            + (cue_inp[:, t, :] * self.str_mask)
                            + (perturb_hid * self.alm_ramp_mask)
                        ))

                hn_next = F.relu(xn_next)
            
            else:

                xn_next = (xn_next 
                            + self.t_const * (
                            -xn_next
                            + (W_rec @ hn_next.T).T
                            + iti_input
                            + cue_inp[:, t, :] * self.str_mask
                            + (perturb_hid * self.alm_ramp_mask)
                        ))

                hn_next = F.relu(xn_next + inhib_stim[:, t, :])

            # append activity to list
            new_xs.append(xn_next)
            new_hs.append(hn_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        pre_out = torch.stack(new_xs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)
        xn_last = new_xs[-1].unsqueeze(0)

        return hn_last, xn_last, rnn_out


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

    def forward(self, inp, hn, inhib_stim, noise=True):

        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''

        # Saving hidden states
        hn_next = hn.squeeze(0)
        size = inp.shape[1]
        new_hs = []

        if self.constrained:

            str2str = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, 1e-15, 1) + self.str2str_fixed) @ self.str2str_D
            str2alm = F.hardtanh(self.str2alm_weight_l0_hh, 1e-15, 1)
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-15, 1) @ self.alm2alm_D
            alm2str = self.alm2str_mask * F.hardtanh(self.alm2str_weight_l0_hh, 1e-15, 1)
            inp_weight = F.hardtanh(self.inp_weight, 1e-15, 1)

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

            if self.constrained:

                hn_next = F.relu(hn_next + 
                        self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ inp_weight * self.str_d1_mask) + inhib_stim[:, t, :]) 
                        + perturb_hid)
            
            else:

                hn_next = F.relu(hn_next + 
                        self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ self.inp_weight * self.str_d1_mask) + inhib_stim[:, t, :]) 
                        + perturb_hid)

            new_hs.append(hn_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)

        return hn_last, rnn_out


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
            0.7 * torch.ones(size=(hid_dim,)),
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
        nn.init.uniform_(self.inp_weight, 0, 0.01)

        # Zeros for no weights
        self.zeros = torch.zeros(size=(hid_dim, hid_dim)).cuda()

        # Time constants for networks (not sure what would be biologically plausible?)
        self.t_const = 0.01

        # Noise level
        self.sigma_recur = noise_level
        self.sigma_input = noise_level

    def forward(self, inp, hn, inhib_stim, noise=True):

        '''
            Forward pass through the model
            
            Parameters:
                inp: input sequence, should be scalar values denoting the target time
                hn: the hidden state of the model
                x: hidden state before activation
        '''

        # Saving hidden states
        hn_next = hn.squeeze(0)
        size = inp.shape[1]
        new_hs = []

        if self.constrained:

            str2str = (self.str2str_mask * F.hardtanh(self.str2str_weight_l0_hh, 1e-15, 1) + self.str2str_fixed) @ self.str2str_D
            str2snr = F.hardtanh(self.str2snr_weight_l0_hh, 1e-15, 1) @ self.str2snr_D
            snr2thal = F.hardtanh(self.snr2thal_weight_l0_hh, 1e-15, 1) @ self.snr2thal_D
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-15, 1) @ self.alm2alm_D
            alm2str = self.alm2str_mask * F.hardtanh(self.alm2str_weight_l0_hh, 1e-15, 1)
            thal2str = self.thal2str_mask * F.hardtanh(self.thal2str_weight_l0_hh, 1e-15, 1)
            thal2alm = F.hardtanh(self.thal2alm_weight_l0_hh, 1e-15, 1)
            inp_weight = F.hardtanh(self.inp_weight, 1e-15, 1)

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
                        self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ inp_weight * self.strthal_mask) + inhib_stim[:, t, :] + self.tonic_inp) 
                        + perturb_hid)
            
            else:

                hn_next = F.relu(hn_next + 
                        self.t_const * (-hn_next + (W_rec @ hn_next.T).T + ((inp[:, t, :] + perturb_inp) @ self.inp_weight * self.strthal_mask) + inhib_stim[:, t, :]) 
                        + perturb_hid)

            new_hs.append(hn_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)

        return hn_last, rnn_out