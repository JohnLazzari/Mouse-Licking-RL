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

        self.alm_ramp_mask = torch.cat([
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(hid_dim,)),
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 4,)),
            torch.ones(size=(hid_dim - int(hid_dim * 0.3),)),
            torch.zeros(size=(int(hid_dim * 0.3),)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.alm_inhib_mask = torch.cat([
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 4,)),
            torch.zeros(size=(hid_dim - int(hid_dim * 0.3),)),
            torch.ones(size=(int(hid_dim * 0.3),)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.full_alm_mask = torch.cat([
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 4,)),
            torch.ones(size=(hid_dim,)),
            torch.ones(size=(inp_dim,)),
        ]).cuda()

        self.iti_mask = torch.cat([
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(hid_dim,)), 
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 5,)),
            torch.ones(size=(inp_dim,)),
        ]).cuda()

        self.str_mask = torch.cat([
            torch.ones(size=(hid_dim,)), 
            torch.ones(size=(hid_dim,)), 
            torch.ones(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 5,)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.str_d1_mask = torch.cat([
            torch.ones(size=(hid_dim,)), 
            torch.zeros(size=(hid_dim,)),
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 5,)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.integrator_mask = torch.cat([
            torch.ones(size=(int(hid_dim/2),)), 
            torch.ones(size=(int(hid_dim/2),)), 
            torch.ones(size=(int(hid_dim/2),)), 
            torch.ones(size=(int(hid_dim/2),)), 
            torch.ones(size=(int(self.fsi_size/2),)),
            torch.ones(size=(int(self.fsi_size/2),)),
            torch.zeros(size=(int(hid_dim/2),)), 
            torch.zeros(size=(int(hid_dim/2),)), 
            torch.zeros(size=(int(hid_dim/2),)), 
            torch.zeros(size=(int(hid_dim/2),)), 
            torch.zeros(size=(hid_dim * 2,)),
            torch.zeros(size=(hid_dim,)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.str_d2_mask = torch.cat([
            torch.zeros(size=(hid_dim,)), 
            torch.ones(size=(hid_dim,)),
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 5,)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.thal_mask = torch.cat([
            torch.zeros(size=(hid_dim,)),
            torch.zeros(size=(hid_dim,)),
            torch.zeros(size=(self.fsi_size,)),
            torch.zeros(size=(hid_dim * 3,)),
            torch.ones(size=(hid_dim,)),
            torch.zeros(size=(hid_dim,)),
            torch.zeros(size=(inp_dim,)),
        ]).cuda()

        self.tonic_inp_d1 = 0.01 * torch.ones(size=(hid_dim,),                device="cuda")
        self.tonic_inp_d2 = 0.01 * torch.ones(size=(hid_dim,),                device="cuda")
        self.tonic_inp_fsi = 0.01 * torch.ones(size=(self.fsi_size,),         device="cuda")
        self.tonic_inp_gpe = 0.8 * torch.ones(size=(hid_dim,),                device="cuda")
        self.tonic_inp_stn = 0.6 * torch.ones(size=(hid_dim,),                device="cuda")
        self.tonic_inp_snr = 0.8 * torch.ones(size=(hid_dim,),          device="cuda")
        self.tonic_inp_thal_int = 0.01 * torch.ones(size=(int(hid_dim/2),),    device="cuda")
        self.tonic_inp_thal_alm = 0.01 * torch.ones(size=(int(hid_dim/2),),    device="cuda")
        self.tonic_inp_alm = 0.01 * torch.ones(size=(hid_dim,),               device="cuda")
        self.tonic_inp_iti = torch.zeros(size=(inp_dim,),               device="cuda")

        self.tonic_inp = torch.cat([
            self.tonic_inp_d1,
            self.tonic_inp_d2,
            self.tonic_inp_fsi,
            self.tonic_inp_gpe,
            self.tonic_inp_stn,
            self.tonic_inp_snr,
            self.tonic_inp_thal_int,
            self.tonic_inp_thal_alm,
            self.tonic_inp_alm,
            self.tonic_inp_iti,
        ])

        ############################
        #   Striatal Recurrence    #
        ############################

        # Inhibitory Connections
        self.d12d1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.d22d2_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.d12d2_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.d22d1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        ############################
        #   Thalamic Projections   #
        ############################

        # Excitatory Connections
        self.thal2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2d1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.thal2d2_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        ############################
        #     ALM Projections      #
        ############################

        # Mix of Excitatory and Inhibitory Connections
        self.alm2alm_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.alm2d1_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.alm2d2_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        ############################
        #     D1 Projections       #
        ############################

        # Inhibitory Connections
        self.d12snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        ############################
        #       D2 Pathway         #
        ############################

        # Inhibitory Connections
        self.d22gpe_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Inhibitory Connections
        self.gpe2stn_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        # Excitatory Connections
        self.stn2snr_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        
        ############################
        #     SNr Projections      #
        ############################

        # Inhibitory Connections
        self.snr2thal_weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))

        ############################
        #     FSI Connections      #
        ############################

        # Inhibitory Connections
        self.fsi2d1_weight = nn.Parameter(torch.empty(size=(hid_dim, self.fsi_size)))
        # Inhibitory Connections
        self.fsi2d2_weight = nn.Parameter(torch.empty(size=(hid_dim, self.fsi_size)))
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
            nn.init.uniform_(self.d12d1_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.d22d2_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.d12d2_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.d22d1_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.thal2alm_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.thal2d1_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.thal2d2_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.alm2alm_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.alm2d1_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.alm2d2_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.d12snr_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.d22gpe_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.gpe2stn_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.stn2snr_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.snr2thal_weight_l0_hh, 0, 1e-2)
            nn.init.uniform_(self.fsi2d1_weight, 0, 1e-2)
            nn.init.uniform_(self.fsi2d2_weight, 0, 1e-2)
            nn.init.uniform_(self.thal2fsi_weight, 0, 1e-2)
            nn.init.uniform_(self.alm2fsi_weight, 0, 1e-2)
            nn.init.uniform_(self.iti2fsi_weight, 0, 1e-2)
            nn.init.uniform_(self.fsi2fsi_weight, 0, 1e-2)

            # Implement Necessary Masks
            # Striatum recurrent weights
            sparse_matrix = torch.empty_like(self.d12d1_weight_l0_hh)
            nn.init.sparse_(sparse_matrix, 0.9)
            self.str2str_sparse_mask = nn.Parameter(torch.where(sparse_matrix != 0, 1, 0), requires_grad=False).cuda()
            self.str2str_D = -1 * torch.eye(hid_dim).cuda()

            self.alm2alm_D = torch.eye(hid_dim).cuda()
            self.alm2alm_D[hid_dim-int(0.3*hid_dim):, 
                            hid_dim-int(0.3*hid_dim):] *= -1
            
            # ALM to striatum weights
            self.alm_mask_excitatory = torch.ones(size=(hid_dim, hid_dim - int(0.3*hid_dim)))
            self.alm_mask_inhibitory = torch.zeros(size=(hid_dim, int(0.3*hid_dim)))
            
            self.alm2str_mask = torch.cat([
                self.alm_mask_excitatory, 
                self.alm_mask_inhibitory
            ], dim=1).cuda()

            # ALM to FSI
            self.alm_mask_excitatory_fsi = torch.ones(size=(self.fsi_size, hid_dim - int(0.3*hid_dim)))
            self.alm_mask_inhibitory_fsi = torch.zeros(size=(self.fsi_size, int(0.3*hid_dim)))

            self.alm2fsi_mask = torch.cat([
                self.alm_mask_excitatory_fsi, 
                self.alm_mask_inhibitory_fsi
            ], dim=1).cuda()
            
            # D1 to SNR D
            self.d12snr_D = -1 * torch.eye(hid_dim).cuda()
            
            # SNR to Thal D
            self.snr2thal_D = -1 * torch.eye(hid_dim).cuda()

            # D2 to GPE D
            self.d22gpe_D = -1 * torch.eye(hid_dim).cuda()
            self.d22gpe_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1)
            ], dim=0).cuda()
            
            self.gpe2stn_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1)
            ], dim=0).cuda()

            self.stn2snr_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1)
            ], dim=0).cuda()

            self.d12snr_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1)
            ], dim=0).cuda()

            self.snr2thal_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(hid_dim/2))),
                    torch.ones(size=(int(hid_dim/2), int(hid_dim/2)))
                ], dim=1)
            ], dim=0).cuda()

            self.thal2alm_mask = torch.cat([
                torch.zeros(size=(hid_dim, int(hid_dim/2))),
                torch.ones(size=(hid_dim, int(hid_dim/2))),
            ], dim=1).cuda()

            # GPE to STN D
            self.gpe2stn_D = -1 * torch.eye(hid_dim).cuda()

            # FSI to STR D
            self.fsi2str_D = -1 * torch.eye(self.fsi_size).cuda()

            # FSI to FSI D
            self.fsi2fsi_D = -1 * torch.eye(self.fsi_size).cuda()
            
            # Thal 2 STR mask
            self.thal2d1_mask = torch.cat([
                torch.zeros(size=(int(hid_dim/2), hid_dim)), 
                torch.ones(size=(int(hid_dim/2), hid_dim)), 
            ], dim=0).cuda()
            
            self.thal2d2_mask = torch.cat([
                torch.zeros(size=(int(hid_dim/2), hid_dim)), 
                torch.ones(size=(int(hid_dim/2), hid_dim)), 
            ], dim=0).cuda()

            self.thal2fsi_mask = torch.cat([
                torch.zeros(size=(int(self.fsi_size/2), hid_dim)), 
                torch.ones(size=(int(self.fsi_size/2), hid_dim)), 
            ], dim=0).cuda()

            self.alm_ramp_2_d1_mask = torch.cat([
                torch.ones(size=(int(hid_dim/2), hid_dim)), 
                torch.zeros(size=(int(hid_dim/2), hid_dim)), 
            ], dim=0).cuda()

            self.alm_ramp_2_d2_mask = torch.cat([
                torch.ones(size=(int(hid_dim/2), hid_dim)), 
                torch.zeros(size=(int(hid_dim/2), hid_dim)), 
            ], dim=0).cuda()

            self.iti_2_d1_mask = torch.cat([
                torch.zeros(size=(int(hid_dim/2), inp_dim)), 
                torch.ones(size=(int(hid_dim/2), inp_dim)), 
            ], dim=0).cuda()

            self.iti_2_d2_mask = torch.cat([
                torch.zeros(size=(int(hid_dim/2), inp_dim)), 
                torch.ones(size=(int(hid_dim/2), inp_dim)), 
            ], dim=0).cuda()

            self.alm_ramp_2_fsi_mask = torch.cat([
                torch.ones(size=(int(self.fsi_size/2), hid_dim)), 
                torch.zeros(size=(int(self.fsi_size/2), hid_dim))
            ], dim=0).cuda()

            self.iti_2_fsi_mask = torch.cat([
                torch.zeros(size=(int(self.fsi_size/2), inp_dim)), 
                torch.ones(size=(int(self.fsi_size/2), inp_dim))
            ], dim=0).cuda()

            self.fsi_2_d1_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(self.fsi_size/2))),
                    torch.zeros(size=(int(hid_dim/2), int(self.fsi_size/2)))
                    ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(self.fsi_size/2))),
                    torch.ones(size=(int(hid_dim/2), int(self.fsi_size/2)))
                    ], dim=1), 
            ], dim=0).cuda()

            self.fsi_2_d2_mask = torch.cat([
                torch.cat([
                    torch.ones(size=(int(hid_dim/2), int(self.fsi_size/2))),
                    torch.zeros(size=(int(hid_dim/2), int(self.fsi_size/2)))
                    ], dim=1), 
                torch.cat([
                    torch.zeros(size=(int(hid_dim/2), int(self.fsi_size/2))),
                    torch.ones(size=(int(hid_dim/2), int(self.fsi_size/2)))
                    ], dim=1), 
            ], dim=0).cuda()

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
        self.inp_weight_d1 = nn.Parameter(torch.empty(size=(hid_dim, inp_dim)))
        nn.init.uniform_(self.inp_weight_d1, 0, 1e-2)

        self.inp_weight_d2 = nn.Parameter(torch.empty(size=(hid_dim, inp_dim)))
        nn.init.uniform_(self.inp_weight_d2, 0, 1e-2)
        
        # Zeros for no weights
        self.zeros = torch.zeros(size=(hid_dim, hid_dim), device="cuda")
        self.zeros_to_iti = torch.zeros(size=(inp_dim, hid_dim), device="cuda")
        self.zeros_from_iti = torch.zeros(size=(hid_dim, inp_dim), device="cuda")
        self.zeros_rec_iti = torch.zeros(size=(inp_dim, inp_dim), device="cuda")

        self.zeros_to_fsi = torch.zeros(size=(self.fsi_size, hid_dim), device="cuda")
        self.zeros_to_fsi_iti = torch.zeros(size=(self.fsi_size, inp_dim), device="cuda")
        self.zeros_rec_fsi = torch.zeros(size=(self.fsi_size, self.fsi_size), device="cuda")
        self.zeros_from_fsi = torch.zeros(size=(hid_dim, self.fsi_size), device="cuda")
        self.zeros_from_fsi2iti = torch.zeros(size=(inp_dim, self.fsi_size), device="cuda")

        # Time constants for networks
        self.t_const = 0.1

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
            d12d1 = (self.str2str_sparse_mask * F.hardtanh(self.d12d1_weight_l0_hh, 1e-10, 1)) @ self.str2str_D
            d22d2 = (self.str2str_sparse_mask * F.hardtanh(self.d22d2_weight_l0_hh, 1e-10, 1)) @ self.str2str_D
            d12d2 = (self.str2str_sparse_mask * F.hardtanh(self.d12d2_weight_l0_hh, 1e-10, 1)) @ self.str2str_D
            d22d1 = (self.str2str_sparse_mask * F.hardtanh(self.d22d1_weight_l0_hh, 1e-10, 1)) @ self.str2str_D
            alm2alm = F.hardtanh(self.alm2alm_weight_l0_hh, 1e-10, 1) @ self.alm2alm_D
            alm2d1 = self.alm2str_mask * F.hardtanh(self.alm2d1_weight_l0_hh, 1e-10, 1)
            alm2d2 = self.alm2str_mask * F.hardtanh(self.alm2d2_weight_l0_hh, 1e-10, 1)
            thal2alm = F.hardtanh(self.thal2alm_weight_l0_hh, 1e-10, 1)
            thal2d1 = F.hardtanh(self.thal2d1_weight_l0_hh, 1e-10, 1)
            thal2d2 = F.hardtanh(self.thal2d2_weight_l0_hh, 1e-10, 1)
            d12snr = F.hardtanh(self.d12snr_weight_l0_hh, 1e-10, 1) @ self.d12snr_D
            d22gpe = F.hardtanh(self.d22gpe_weight_l0_hh, 1e-10, 1) @ self.d22gpe_D
            gpe2stn = F.hardtanh(self.gpe2stn_weight_l0_hh, 1e-10, 1) @ self.gpe2stn_D
            stn2snr = F.hardtanh(self.stn2snr_weight_l0_hh, 1e-10, 1)
            snr2thal = F.hardtanh(self.snr2thal_weight_l0_hh, 1e-10, 1) @ self.snr2thal_D
            fsi2d1 = F.hardtanh(self.fsi2d1_weight, 1e-10, 1) @ self.fsi2str_D
            fsi2d2 = F.hardtanh(self.fsi2d2_weight, 1e-10, 1) @ self.fsi2str_D
            thal2fsi = F.hardtanh(self.thal2fsi_weight, 1e-10, 1)
            alm2fsi = self.alm2fsi_mask * F.hardtanh(self.alm2fsi_weight, 1e-10, 1)
            iti2fsi = F.hardtanh(self.iti2fsi_weight, 1e-10, 1)
            fsi2fsi = F.hardtanh(self.fsi2fsi_weight, 1e-10, 1) @ self.fsi2fsi_D
            inp_weight_d1 = F.hardtanh(self.inp_weight_d1, 1e-10, 1)
            inp_weight_d2 = F.hardtanh(self.inp_weight_d2, 1e-10, 1)

            # Concatenate into single weight matrix

                            #  D1      D2     FSI         GPE        STN         SNR       Thal    ALM         ALM ITI
            W_d1 = torch.cat([d12d1, d22d1, fsi2d1, self.zeros, self.zeros, self.zeros, thal2d1, alm2d1, inp_weight_d1],                                                                                                dim=1)      # D1
            W_d2 = torch.cat([d12d2, d22d2, fsi2d2, self.zeros, self.zeros, self.zeros, thal2d2, alm2d2, inp_weight_d2],                                                                                                dim=1)      # D2
            W_fsi = torch.cat([self.zeros_to_fsi, self.zeros_to_fsi, fsi2fsi, self.zeros_to_fsi, self.zeros_to_fsi, self.zeros_to_fsi, thal2fsi, alm2fsi, iti2fsi],                                                     dim=1)      # FSI
            W_gpe = torch.cat([self.zeros, d22gpe, self.zeros_from_fsi, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros_from_iti],                                                               dim=1)      # GPE
            W_stn = torch.cat([self.zeros, self.zeros, self.zeros_from_fsi, gpe2stn, self.zeros, self.zeros, self.zeros, self.zeros, self.zeros_from_iti],                                                              dim=1)      # STN
            W_snr = torch.cat([d12snr, self.zeros, self.zeros_from_fsi, self.zeros, stn2snr, self.zeros, self.zeros, self.zeros, self.zeros_from_iti],                                                                  dim=1)      # SNR
            W_thal = torch.cat([self.zeros, self.zeros, self.zeros_from_fsi, self.zeros, self.zeros, snr2thal, self.zeros, self.zeros, self.zeros_from_iti],                                                            dim=1)      # Thal
            W_alm = torch.cat([self.zeros, self.zeros, self.zeros_from_fsi, self.zeros, self.zeros, self.zeros, thal2alm, alm2alm, self.zeros_from_iti],                                                                dim=1)      # ALM
            W_alm_iti = torch.cat([self.zeros_to_iti, self.zeros_to_iti, self.zeros_from_fsi2iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_to_iti, self.zeros_rec_iti],   dim=1)      # ITI

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
        W_rec = torch.cat([W_d1, W_d2, W_fsi, W_gpe, W_stn, W_snr, W_thal, W_alm, W_alm_iti], dim=0)

        #plt.imshow(W_rec.detach().cpu().numpy())
        #plt.show()

        # Add noise to the system if specified
        if noise:

            means = {}
            stds = {}

            zeros_pre_cue = torch.zeros(size=(4, 100, 1))
            zeros_post_lick = torch.zeros(size=(230, 1))

            means[0] = torch.zeros(size=(80, 1))
            means[1] = torch.zeros(size=(110, 1))
            means[2] = torch.zeros(size=(140, 1))
            means[3] = torch.cat([
                torch.zeros(size=(170, 1)),
                zeros_post_lick
            ], dim=0)

            stds[0] = torch.ones(size=(80, 1))
            stds[1] = torch.ones(size=(110, 1))
            stds[2] = torch.ones(size=(140, 1))
            stds[3] = torch.cat([
                torch.ones(size=(170, 1)),
                zeros_post_lick
            ], dim=0)

            means_hid = pad_sequence([means[0], means[1], means[2], means[3]], batch_first=True)
            stds_hid = pad_sequence([stds[0], stds[1], stds[2], stds[3]], batch_first=True)

            perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * torch.normal(means_hid, stds_hid)
            perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * torch.normal(means_hid, stds_hid)

            perturb_hid = torch.cat([zeros_pre_cue, perturb_hid], dim=1).cuda()
            perturb_inp = torch.cat([zeros_pre_cue, perturb_inp], dim=1).cuda()

        else:

            perturb_hid = torch.zeros([4, 1000, 1]).cuda()
            perturb_inp = torch.zeros([4, 1000, 1]).cuda()

        # Loop through RNN
        for t in range(size):

            # Get the ITI mode input to the network
            iti_act = inp[:, t, :] + perturb_inp[:, t, :]
            non_iti_mask = torch.zeros(size=(iti_act.shape[0], self.hid_dim * 7 + self.fsi_size), device="cuda")
            iti_input = torch.cat([non_iti_mask, iti_act], dim=-1)

            # Get the activity of the next hidden state
            if self.constrained:

                xn_next = (xn_next 
                            + self.t_const * (
                            -xn_next
                            + (W_rec @ hn_next.T).T
                            + iti_input
                            + self.tonic_inp
                            + inhib_stim[:, t, :]
                            + (cue_inp[:, t, :] * self.thal_mask)
                            + (perturb_hid[:, t, :])
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