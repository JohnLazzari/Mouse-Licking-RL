import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt

class Region(nn.Module):
    def __init__(
        self,
        name,
        num_units,
        base_firing,
        device,
        cell_types=None
    ):
        super(Region, self).__init__()
        
        self.name = name
        self.num_units = num_units
        self.device = device
        self.connections = {}
        self.base_firing = base_firing * torch.ones(size=(num_units,))
        self.cell_type_info = cell_types
        self.masks = {}

        self.generate_masks()
    
    def add_connection(
        self,
        proj_region,
        fan_out,
        weight_mask=None,
        fixed_weights=None,
        sign_matrix=None,
        lower_bound=0,
        upper_bound=1e-2
    ):

        connection_properties = {}

        # implement connections
        parameter = nn.Parameter(torch.empty(size=(fan_out, self.num_units)))
        nn.init.uniform_(parameter, lower_bound, upper_bound)

        # Trainable values
        connection_properties["parameter"] = parameter.to(self.device)

        # Masks applied on weights
        if weight_mask == None:
            
            weight_mask = torch.ones_like(parameter)
        
        if fixed_weights == None:
            
            fixed_weights = torch.zeros_like(parameter)

        if sign_matrix == None:
            
            sign_matrix = torch.eye(self.num_units)

        connection_properties["weight_mask"] = weight_mask
        connection_properties["fixed_weights"] = fixed_weights
        connection_properties["sign_matrix"] = sign_matrix

        # Output size of weight
        connection_properties["fan_out"] = fan_out

        self.connections[proj_region] = connection_properties
    
    def _generate_masks(
        self
    ):

        full_mask = torch.ones(size=(self.num_units))
        zeros_mask = torch.zeros(size=(self.num_units))
        self.masks["full"] = full_mask
        self.masks["zeros"] = zeros_mask

        for key in self.cell_type_info:
            
            cur_masks = []
            for prev_key in self.cell_type_info:
                
                if prev_key == key:
                    
                    cur_masks.append(torch.ones(size=(int(self.num_units * self.cell_type_info[prev_key]))))
                
                else:
                    
                    cur_masks.append(torch.zeros(size=(int(self.num_units * self.cell_type_info[prev_key]))))

            self.masks[key] = torch.cat(cur_masks)
    
    def has_connection_to(
        self,
        region
    ):
        
        for key in self.connections:
            
            if key == region.name:
                
                return True 
        
        return False

class mRNN(nn.Module):
    def __init__(
        self, 
        inp_dim, 
        out_dim, 
        noise_level_act=0.01, 
        noise_level_inp=0.01, 
        constrained=True, 
        out_type="simple"):

        super(mRNN, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                inp_dim: dimension of input
                hid_dim: number of hidden neurons in each region
                action_dim: output dimension, should be one for lick or no lick
        '''

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.constrained = constrained
        self.region_list = {}

        str_cell_types = {
            "d1": 0.5,
            "d2": 0.5
        }

        alm_cell_types = {
            "exc": 0.7,
            "inhib": 0.3
        }

        # Generate regions
        striatum = Region(
            "striatum", 
            256, 
            0, 
            "cuda",
            str_cell_types
        )

        gpe = Region(
            "gpe", 
            256, 
            .8, 
            "cuda"
        )

        stn = Region(
            "stn", 
            256, 
            .5, 
            "cuda"
        )

        snr = Region(
            "snr", 
            256, 
            .8, 
            "cuda"
        )

        thal = Region(
            "thal", 
            256, 
            .8, 
            "cuda"
        )

        alm = Region(
            "alm", 
            256, 
            0, 
            "cuda",
            alm_cell_types
        )

        iti = Region(
            "iti", 
            int(256 * 0.3), 
            0, 
            "cuda"
        )

        self.region_list["striatum"] = striatum
        self.region_list["gpe"] = gpe
        self.region_list["stn"] = stn
        self.region_list["snr"] = snr
        self.region_list["thal"] = thal
        self.region_list["alm"] = alm
        self.region_list["iti"] = iti

        ############################
        #   Striatal Connections   #
        ############################

        sparse_matrix = torch.empty(size=(striatum.num_units, striatum.num_units))
        nn.init.sparse_(sparse_matrix, 0.9)

        str2str_sparse_mask = nn.Parameter(torch.where(sparse_matrix != 0, 1, 0), requires_grad=False).cuda()
        str2str_D = -1 * torch.eye(striatum.num_units).cuda()

        str2snr_mask = torch.cat([
            torch.ones(size=(snr.num_units, int(striatum.num_units * 0.5))),
            torch.zeros(size=(snr.num_units, int(striatum.num_units * 0.5)))
        ], dim=1)

        str2snr_D = -1 * torch.eye(striatum.num_units)

        str2gpe_mask = torch.cat([
            torch.zeros(size=(gpe.num_units, int(striatum.num_units * 0.5))),
            torch.ones(size=(gpe.num_units, int(striatum.num_units * 0.5)))
        ], dim=1)

        str2gpe_D = -1 * torch.eye(striatum.num_units)

        # Inhibitory Connections
        striatum.add_connection(
            "striatum",
            striatum.num_units,
            weight_mask=str2str_sparse_mask,
            sign_matrix=str2str_D
        )

        striatum.add_connection(
            "snr",
            snr.num_units,
            weight_mask=str2snr_mask,
            sign_matrix=str2snr_D
        )

        striatum.add_connection(
            "gpe",
            gpe.num_units,
            weight_mask=str2gpe_mask,
            sign_matrix=str2gpe_D
        )

        ############################
        #   Thalamic Projections   #
        ############################

        thal.add_connection(
            "striatum",
            striatum.num_units,
        )

        thal.add_connection(
            "alm",
            alm.num_units,
        )

        ############################
        #     ALM Projections      #
        ############################

        alm2alm_D = torch.eye(alm.num_units).cuda()
        alm2alm_D[:, alm.num_units-int(0.3*alm.num_units):] *= -1

        alm.add_connection(
            "alm",
            alm.num_units,
            sign_matrix=alm2alm_D
        )

        alm_mask_excitatory = torch.ones(size=(alm.num_units, alm.num_units - int(0.3*alm.num_units)))
        alm_mask_inhibitory = torch.zeros(size=(alm.num_units, int(0.3*alm.num_units)))
        
        alm2str_mask = torch.cat([
            alm_mask_excitatory, 
            alm_mask_inhibitory
        ], dim=1).cuda()

        alm.add_connection(
            "striatum",
            striatum.num_units,
            weight_mask=alm2str_mask
        )

        ############################
        #       D2 Pathway         #
        ############################

        gpe2stn_D = -1 * torch.eye(gpe.num_units).cuda()

        gpe.add_connection(
            "stn",
            stn.num_units,
            sign_matrix=gpe2stn_D 
        )

        stn.add_connection(
            "snr",
            snr.num_units
        )

        ############################
        #     SNr Projections      #
        ############################

        snr2thal_D = -1 * torch.eye(snr.num_units).cuda()

        snr.add_connection(
            "thal",
            thal.num_units,
            sign_matrix=snr2thal_D
        )

        ############################
        #     ITI Projections      #
        ############################

        iti.add_connection(
            "striatum",
            striatum.num_units
        )

        # Output weights
        if out_type == "simple":

            self.out_weight_alm = (1 / alm.num_units) * torch.ones(size=(out_dim, self.alm_exc_size))

        elif out_type == "data":

            self.out_weight_alm = nn.Parameter(torch.empty(size=(out_dim, self.alm_exc_size)))
            nn.init.uniform_(self.out_weight_alm, 0, 1)
        
        # Time constants for networks
        self.t_const = 0.1

        # Noise level
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        
        self.W_rec, self.W_rec_mask, self.W_rec_fixed, self.W_rec_sign_matrix = self.gen_w_rec()

        self.total_num_units = self._get_total_num_units()
        self.tonic_inp = self._get_tonic_inp()

        self.out_weight_alm = nn.Parameter(size=(inp_dim, int(alm.num_units * alm.cell_type_info["exc"])))

        self.alm_start_idx, self.alm_end_idx = self.get_region_indices()


    def _get_total_num_units(
        self
    ):
        
        units = 0
        for region in self.region_list:
            units = units + self.region_list[region].num_units
        return units

    
    def _get_tonic_inp(
        self
    ):
        
        tonic_inp = []
        for region in self.region_list.values():
            tonic_inp.append(region.base_firing)
        return torch.cat(tonic_inp)


    def gen_w_rec(
        self
    ):
        
        region_connection_columns = []
        region_weight_mask_columns = []
        region_fixed_weights_columns = []
        region_sign_matrix_columns = []

        for region in self.region_list.values():

            self._get_full_connectivity(region)

            connections_from_region = []
            weight_mask_from_region = []
            fixed_weights_from_region = []
            sign_matrix_from_region = []

            for connection in region.connections:

                connections_from_region.append(connection["parameter"])
                weight_mask_from_region.append(connection["weight_mask"])
                fixed_weights_from_region.append(connection["fixed_weights"])
                sign_matrix_from_region.append(connection["sign_matrix"])
            
            full_connections = torch.cat(connections_from_region, dim=0)
            full_weight_mask = torch.cat(weight_mask_from_region, dim=0)
            full_fixed_weights = torch.cat(fixed_weights_from_region, dim=0)
            full_sign_matrix = torch.cat(sign_matrix_from_region, dim=0)

            region_connection_columns.append(full_connections)
            region_weight_mask_columns.append(full_weight_mask)
            region_fixed_weights_columns.append(full_fixed_weights)
            region_sign_matrix_columns.append(full_sign_matrix)
        
        W_rec = torch.cat(region_connection_columns, dim=1)
        W_rec_mask = torch.cat(region_weight_mask_columns, dim=1)
        W_rec_fixed = torch.cat(region_fixed_weights_columns, dim=1)
        W_rec_sign = torch.cat(region_sign_matrix_columns, dim=1)

        return W_rec, W_rec_mask, W_rec_fixed, W_rec_sign
            
            
    def _get_full_connectivity(
        self,
        region,
        region_list
    ):

        for other_region in region_list.values():
            
            if region.has_connection_to(other_region):

                continue

            weight_mask = torch.zeros(size=(other_region.num_units, region.num_units))

            region.add_connection(
                other_region.name,
                other_region.num_units,
                weight_mask=weight_mask
            )
    

    def apply_dales_law(
        self
    ):
        
        W_rec = (self.W_rec_mask * F.relu(self.W_rec) + self.W_rec_fixed) @ self.W_rec_sign_matrix
        return W_rec
    

    def get_region_indices(
        self,
        region
    ):
        
        start_idx = 0
        end_idx = 0
        for cur_reg in self.region_list:
            if cur_reg == region:
                end_idx = start_idx + self.region_list[cur_reg].num_units
                break
            start_idx = start_idx + self.region_list[cur_reg].num_units
        return start_idx, end_idx


    def forward(
        self, 
        inp, 
        cue_inp, 
        hn, 
        xn, 
        inhib_stim, 
        noise=True
    ):

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
        outs = []

        if self.constrained:

            W_rec = self.apply_dales_law
            out_weight_alm = F.relu(self.out_weight_alm)

        # Add noise to the system if specified
        if noise:

            means = {}
            stds = {}

            zeros_pre_cue = torch.zeros(size=(4, 100, 1))
            zeros_post_lick = torch.zeros(size=(230, 1))

            means[0] = torch.zeros(size=(110, 1))
            means[1] = torch.zeros(size=(140, 1))
            means[2] = torch.zeros(size=(170, 1))
            means[3] = torch.cat([
                torch.zeros(size=(200, 1)),
                zeros_post_lick
            ], dim=0)

            stds[0] = torch.ones(size=(110, 1))
            stds[1] = torch.ones(size=(140, 1))
            stds[2] = torch.ones(size=(170, 1))
            stds[3] = torch.cat([
                torch.ones(size=(200, 1)),
                zeros_post_lick
            ], dim=0)

            means_hid = pad_sequence([means[0], means[1], means[2], means[3]], batch_first=True)
            stds_hid = pad_sequence([stds[0], stds[1], stds[2], stds[3]], batch_first=True)

            perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * torch.normal(means_hid, stds_hid)
            perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * torch.normal(means_hid, stds_hid)

            perturb_hid = torch.cat([zeros_pre_cue, perturb_hid], dim=1).cuda()
            perturb_inp = torch.cat([zeros_pre_cue, perturb_inp], dim=1).cuda()

        else:

            perturb_hid = torch.zeros([1, 1000, 1]).cuda()
            perturb_inp = torch.zeros([1, 1000, 1]).cuda()

        # Loop through RNN
        for t in range(size):

            # Get the ITI mode input to the network
            iti_act = inp[:, t, :] + perturb_inp[:, t, :]
            non_iti_mask = torch.zeros(size=(iti_act.shape[0], self.hid_dim * 7), device="cuda")
            iti_input = torch.cat([non_iti_mask, iti_act], dim=-1)

            # Get the activity of the next hidden state
            if self.constrained:

                xn_next = (xn_next 
                            + self.t_const 
                            * (
                            -xn_next
                            + (W_rec @ hn_next.T).T
                            + iti_input
                            + self.tonic_inp
                            + inhib_stim[:, t, :]
                            + (cue_inp[:, t, :] * self.thal_mask)
                            + (perturb_hid[:, t, :] * self.alm_ramp_mask)
                        ))

                hn_next = F.relu(xn_next)

                alm_hn = hn_next[:, self.alm_start_idx:self.alm_end_idx]

                out = (out_weight_alm @ (alm_hn * self.alm.masks["exc"]).T).T
            
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
            outs.append(out)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)
        alm_out = torch.stack(outs, dim=1)

        return rnn_out, alm_out