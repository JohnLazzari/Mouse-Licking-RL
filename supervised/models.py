import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import json

class Region(nn.Module):
    def __init__(
        self,
        num_units,
        base_firing,
        device,
        cell_types=None
    ):
        super(Region, self).__init__()

        '''
            Region class that represents connections from one region to others among other properties
            
            Params:
                name:               name of region                   
                num_units:          number of neurons in region
                base_firing:        baseline level of firing for region
                device:             cuda or cpu
                cell_types:         dictionary specifying each cell type within the region and the percentage of neurons each type occupies

        '''
        
        self.num_units = num_units
        self.device = device
        self.connections = {}
        self.base_firing = base_firing * torch.ones(size=(num_units,))
        self.cell_type_info = cell_types
        self.masks = {}

        self._generate_masks()
    
    def add_connection(
        self,
        proj_region_name,
        proj_region,
        src_region_cell_type,
        dst_region_cell_type,
        sign,
        lower_bound=0,
        upper_bound=1e-2
    ):

        '''
            Add a connection from current region to specified region
            
            Params:
                proj_region:        Region that connection is projecting to
                fan_out:            Number of nodes in connecting region
                weight_mask:        Mask to zero out specified connections
                fixed_weights:      Matrix added to W_rec specifying fixed connections
                sign_matrix:        Matrix of ones or negative ones denoting the sign of each connection
                lower_bound:        lower bound for uniform initialization
                upper_bound:        upper bound for uniform initialization
        '''

        connection_properties = {}

        # implement connections
        parameter = torch.empty(size=(proj_region.num_units, self.num_units)).uniform_(lower_bound, upper_bound)

        # Trainable values
        connection_properties["parameter"] = parameter.to(self.device)

        # Masks applied on weights
        if src_region_cell_type is not None:
            
            weight_mask_src = torch.ones_like(parameter) * self.masks[src_region_cell_type]
        
        else:
            
            weight_mask_src = torch.ones_like(parameter)
        
        if dst_region_cell_type is not None:
            
            weight_mask_dst = torch.ones_like(parameter) * proj_region.masks[dst_region_cell_type]
        
        else:
            
            weight_mask_dst = torch.ones_like(parameter)
        
        weight_mask = weight_mask_src * weight_mask_dst

        if sign == "exc":

            sign_matrix = torch.ones(size=(proj_region.num_units, self.num_units))
        
        elif sign == "inhib":
            
            sign_matrix = -1 * torch.ones(size=(proj_region.num_units, self.num_units))

        connection_properties["weight_mask"] = weight_mask.to(self.device)
        connection_properties["sign_matrix"] = sign_matrix.to(self.device)

        self.connections[proj_region_name] = connection_properties
    
    def _generate_masks(
        self
    ):

        '''
            Generate masks for hn including the full region and different cell types
        '''

        full_mask = torch.ones(size=(self.num_units,))
        zero_mask = torch.zeros(size=(self.num_units,))

        self.masks["full"] = full_mask
        self.masks["zero"] = zero_mask

        if self.cell_type_info is not None:

            for key in self.cell_type_info:
                
                cur_masks = []

                for prev_key in self.cell_type_info:
                    
                    if prev_key == key:
                        
                        cur_masks.append(torch.ones(size=(int(self.num_units * self.cell_type_info[prev_key]),)))
                    
                    else:
                        
                        cur_masks.append(torch.zeros(size=(int(self.num_units * self.cell_type_info[prev_key]),)))

                self.masks[key] = torch.cat(cur_masks).to(self.device)

    
    def has_connection_to(
        self,
        region
    ):

        '''
            Check if there is a connection between the current region and the specified region
            
            Params:
                region:     region used to test connection with
        '''
        
        for key in self.connections:
            
            if key == region.name:
                
                return True 
        
        return False


class mRNN(nn.Module):
    def __init__(
        self, 
        config,
        inp_dim, 
        noise_level_act=0.01, 
        noise_level_inp=0.01, 
        constrained=True, 
        t_const=0.1,
        device="cuda",
    ):

        super(mRNN, self).__init__()
        
        '''
            Multi-Regional RNN model, implements interaction between striatum and ALM
            
            parameters:
                config:             json configuration file to automatically generate network
                inp_dim:            dimension of input
                out_dim:            output dimension, should be one for lick or no lick
        '''

        # Load the configuration from the JSON file
        with open(config, 'r') as f:
            config = json.load(f)

        # Gather mRNN variables
        self.region_dict = {}
        self.inp_dim = inp_dim
        self.constrained = constrained
        self.device = device

        # Generate a Region class for each region and store them in region_dict
        self.__gen_regions(
            config["regions"]
        )

        # Generate specified connections between regions
        self.__gen_connections(
            config["connections"]
        )

        # Time constants for networks
        self.t_const = t_const

        # Noise level
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp
        
        self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix = self.gen_w_rec()

        self.total_num_units = self.__get_total_num_units()
        self.tonic_inp = self.__get_tonic_inp()

        self.alm_start_idx, self.alm_end_idx = self.get_region_indices("alm")

        self.thal_mask = self.__gen_region_mask("thal")


    def gen_w_rec(
        self
    ):
        
        '''
            Generate the full connectivity matrix W_rec
            Generate the corresponding masks for W_rec
            
            W_rec encompasses all connections between all regions in mRNN
        '''
        
        region_connection_columns = []
        region_weight_mask_columns = []
        region_sign_matrix_columns = []

        for cur_region in self.region_list:

            self.__get_full_connectivity(self.region_list[cur_region])

            connections_from_region = []
            weight_mask_from_region = []
            sign_matrix_from_region = []

            for connection in self.ordering:

                connections_from_region.append(self.region_list[cur_region].connections[connection]["parameter"])
                weight_mask_from_region.append(self.region_list[cur_region].connections[connection]["weight_mask"])
                sign_matrix_from_region.append(self.region_list[cur_region].connections[connection]["sign_matrix"])
            
            full_connections = torch.cat(connections_from_region, dim=0)
            full_weight_mask = torch.cat(weight_mask_from_region, dim=0)
            full_sign_matrix = torch.cat(sign_matrix_from_region, dim=0)

            region_connection_columns.append(full_connections)
            region_weight_mask_columns.append(full_weight_mask)
            region_sign_matrix_columns.append(full_sign_matrix)
        
        W_rec = nn.Parameter(torch.cat(region_connection_columns, dim=1))
        W_rec_mask = torch.cat(region_weight_mask_columns, dim=1)
        W_rec_sign = torch.cat(region_sign_matrix_columns, dim=1)

        return W_rec, W_rec_mask, W_rec_sign


    def apply_dales_law(
        self
    ):

        '''
            Apply formula to ensure Dale's law is preserved on W_rec
        '''
        
        W_rec = self.W_rec_mask * F.relu(self.W_rec) * self.W_rec_sign_matrix
        return W_rec
    

    def get_region_indices(
        self,
        region
    ):
        
        '''
            Get the indices of the specified region in hn
            
            Params:
                region:     Region to gather indices for
        '''
        
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

        if self.constrained:

            W_rec = self.apply_dales_law()

        # Add noise to the system if specified
        if noise:

            perturb_hid = np.sqrt(2*self.t_const*self.sigma_recur**2) * np.normal(0, 1)
            perturb_inp = np.sqrt(2*self.t_const*self.sigma_input**2) * np.normal(0, 1)

        else:

            perturb_hid = 0
            perturb_inp = 0

        # Loop through RNN
        for t in range(size):

            # Get the ITI mode input to the network
            iti_act = inp[:, t, :] + perturb_inp[:, t, :]
            non_iti_mask = torch.zeros(size=(iti_act.shape[0], self.total_num_units - self.region_list["iti"].num_units), device="cuda")
            iti_input = torch.cat([non_iti_mask, iti_act], dim=-1)

            xn_next = (xn_next 
                        + self.t_const 
                        * (
                        -xn_next
                        + (W_rec @ hn_next.T).T
                        + iti_input
                        + self.tonic_inp
                        + inhib_stim[:, t, :]
                        + (cue_inp[:, t, :] * self.thal_mask)
                        + (perturb_hid[:, t, :])
                    ))

            hn_next = F.relu(xn_next)

            # append activity to list
            new_xs.append(xn_next)
            new_hs.append(hn_next)
        
        # Collect hidden states
        rnn_out = torch.stack(new_hs, dim=1)

        return rnn_out


    def __gen_regions(
        self,
        regions
    ):

        '''
            Generate the regions from the inputted regions list and store them in a dictionary
            
            Params:
                regions:        list of dictionaries containing region properties
        '''
        
        for region in regions:
            
            # Instantiate a Region class with specified properties and store with proper name
            self.region_dict[region["name"]] = Region(
                num_units=region["num_units"],
                base_firing=region["base_firing"],
                device=self.device,
                cell_types=region["cell_types"]
            )


    def __gen_connections(
        self,
        connections
    ):

        '''
            Generate the specified connections within each region object
            
            Params:
                connections:        list of dictionaries containing connection properties
        '''
        
        for connection in connections:
            
            self.region_list[connection["src_region"]].add_connection(
                proj_region_name=connection["dst_region"],
                proj_region=self.region_list["dst_region"],
                src_region_cell_type=connection["src_region_cell_type"],
                dst_region_cell_type=connection["dst_region_cell_type"],
                sign=connection["sign"]
            )

    def __gen_region_mask(
        self,
        region,
        cell_type=None
    ):
        
        '''
            Generate a mask for hn that only captures specified region
            
            Params:
                region:         Region to generate mask for
                cell_type:      Specify if the mask corresponds to a particular cell type within the region
        '''

        if cell_type is None:
            mask_type = "full"
        else:
            mask_type = cell_type
        
        mask = []
        for next_region in self.region_list:

            if region == next_region:

                mask.append(self.region_list[region].masks[mask_type])
            
            else:

                mask.append(self.region_list[next_region].masks["zero"])
        
        mask = torch.cat(mask).to(self.device)

        return mask
            
            
    def __get_full_connectivity(
        self,
        region,
    ):
        
        '''
            Gather all of the non-specified connections between specified region and all other regions in mRNN
            Create zero masks for these connections as they denote no connectivity
            
            Params:
                region:     Current region to complete connections in mRNN for
        '''

        for other_region in self.region_list.values():
            
            if region.has_connection_to(other_region):

                continue

            weight_mask = torch.zeros(size=(other_region.num_units, region.num_units))
            fixed_weights = torch.zeros(size=(other_region.num_units, region.num_units))
            sign_matrix = torch.zeros(size=(other_region.num_units, region.num_units))

            region.add_connection(
                other_region.name,
                other_region.num_units,
                weight_mask=weight_mask,
                fixed_weights=fixed_weights,
                sign_matrix=sign_matrix
            )


    def __get_total_num_units(
        self
    ):

        '''
            Count the total number of units in the mRNN
        '''
        
        units = 0
        for region in self.region_list:
            units = units + self.region_list[region].num_units
        return units

    
    def __get_tonic_inp(
        self
    ):
        
        '''
            Gather all regional baseline firing levels into bias vector
        '''
        
        tonic_inp = []
        for region in self.region_list.values():
            tonic_inp.append(region.base_firing)
        return torch.cat(tonic_inp).to(self.device)
    

class CBGTCL(nn.Module):
    def __init__(
        self,
        config,
        inp_dim,
        out_dim
    ):
        super(CBGTCL, self).__init__()

        # Output weights
        if out_type == "ramp":

            self.out_weight_alm = (1 / alm.num_units) * torch.ones(size=(out_dim, alm.num_units)).to(self.device)

        elif out_type == "data":

            self.out_weight_alm = nn.Parameter(torch.empty(size=(out_dim, alm.num_units))).to(self.device)
            nn.init.uniform_(self.out_weight_alm, 0, 1)
        