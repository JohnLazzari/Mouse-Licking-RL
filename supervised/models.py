import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math
import matplotlib.pyplot as plt
import json

import torch
import torch.nn as nn

class Region(nn.Module):
    """
    A class representing a region in a neural network that models connections 
    to other regions along with other properties such as cell types and firing rates.
    
    Attributes:
        num_units (int): Number of neurons in the region.
        base_firing (torch.Tensor): Baseline firing rate for each neuron in the region.
        device (torch.device): The device on which to store the tensors (e.g., 'cpu' or 'cuda').
        cell_type_info (dict): Dictionary specifying each cell type and the proportion of neurons each type occupies.
        connections (dict): Dictionary to store the connections to other regions.
        masks (dict): Masks for each cell type and region properties (e.g., full mask, zero mask).
    """
    
    def __init__(self, num_units, base_firing, device, cell_types=None):
        """
        Initializes the Region class.
        
        Args:
            num_units (int): Number of neurons in the region.
            base_firing (float): Baseline firing rate for the region.
            device (torch.device): The device ('cpu' or 'cuda').
            cell_types (dict, optional): A dictionary specifying the proportions of different cell types in the region.
        """
        super(Region, self).__init__()

        self.num_units = num_units
        self.device = device
        self.base_firing = base_firing * torch.ones(size=(num_units,))
        self.cell_type_info = cell_types if cell_types is not None else {}
        self.connections = {}
        self.masks = {}

        self._generate_masks()

    def add_connection(
        self, 
        proj_region_name, 
        proj_region, 
        src_region_cell_type, 
        dst_region_cell_type, 
        sign, 
        zero_connection=False, 
        lower_bound=0, 
        upper_bound=1e-2
    ):
        """
        Adds a connection from the current region to a specified projection region.
        
        Args:
            proj_region_name (str): Name of the region that the current region connects to.
            proj_region (Region): The target region to which the connection is made.
            src_region_cell_type (str, optional): The source region's cell type.
            dst_region_cell_type (str, optional): The destination region's cell type.
            sign (str, optional): Specifies if the connection is excitatory or inhibitory ('inhib' for inhibitory).
            zero_connection (bool, optional): If True, no connections are created (default is False).
            lower_bound (float, optional): Lower bound for uniform weight initialization.
            upper_bound (float, optional): Upper bound for uniform weight initialization.
        """
        connection_properties = {}

        # Initialize connection parameters
        if not zero_connection:
            parameter = torch.empty(size=(proj_region.num_units, self.num_units)).uniform_(lower_bound, upper_bound)
        else:
            parameter = torch.zeros(size=(proj_region.num_units, self.num_units))

        # Store trainable parameter
        connection_properties["parameter"] = parameter.to(self.device)

        # Initialize connection tensors (1s for active connections, 0s for no connections)
        connection_tensor = torch.ones_like(parameter).to(self.device) if not zero_connection else torch.zeros_like(parameter).to(self.device)

        # Create weight masks based on cell types, if specified
        weight_mask_src, sign_matrix_src = self._get_weight_and_sign_matrices(src_region_cell_type, connection_tensor)
        weight_mask_dst, sign_matrix_dst = proj_region._get_weight_and_sign_matrices(dst_region_cell_type, connection_tensor)

        # Combine masks
        weight_mask = weight_mask_src * weight_mask_dst
        sign_matrix = sign_matrix_src * sign_matrix_dst

        # Adjust the sign matrix for inhibitory connections
        if sign == "inhib":
            sign_matrix *= -1
        elif sign is None:
            sign_matrix = torch.zeros_like(parameter).to(self.device)

        # Store weight mask and sign matrix
        connection_properties["weight_mask"] = weight_mask.to(self.device)
        connection_properties["sign_matrix"] = sign_matrix.to(self.device)

        # Update connections dictionary
        if proj_region_name in self.connections:
            self.connections[proj_region_name]["weight_mask"] += connection_properties["weight_mask"]
            self.connections[proj_region_name]["sign_matrix"] += connection_properties["sign_matrix"]
        else:
            self.connections[proj_region_name] = connection_properties

    def _generate_masks(self):
        """
        Generates masks for the region, including full and zero masks, and specific cell-type masks.
        """
        full_mask = torch.ones(size=(self.num_units,)).to(self.device)
        zero_mask = torch.zeros(size=(self.num_units,)).to(self.device)

        self.masks["full"] = full_mask
        self.masks["zero"] = zero_mask

        for key in self.cell_type_info:
            mask = self._generate_cell_type_mask(key)
            self.masks[key] = mask.to(self.device)

    def _generate_cell_type_mask(self, key):
        """
        Generates a mask for a specific cell type based on its proportion in the region.

        Args:
            key (str): The cell type identifier.

        Returns:
            torch.Tensor: Mask for the specified cell type.
        """
        cur_masks = []
        for prev_key in self.cell_type_info:
            if prev_key == key:
                cur_masks.append(torch.ones(size=(int(round(self.num_units * self.cell_type_info[prev_key])),)))
            else:
                cur_masks.append(torch.zeros(size=(int(round(self.num_units * self.cell_type_info[prev_key])),)))
        mask = torch.cat(cur_masks)
        return mask

    def _get_weight_and_sign_matrices(self, cell_type, connection_tensor):
        """
        Retrieves the weight mask and sign matrix for a specified cell type.

        Args:
            cell_type (str): The cell type to generate the mask for.
            connection_tensor (torch.Tensor): Tensor indicating whether connections are active.

        Returns:
            tuple: weight mask and sign matrix.
        """
        if cell_type is not None:
            weight_mask = connection_tensor * self.masks.get(cell_type, connection_tensor)
            sign_matrix = connection_tensor * self.masks.get(cell_type, connection_tensor)
        else:
            weight_mask = connection_tensor
            sign_matrix = connection_tensor

        return weight_mask, sign_matrix

    def has_connection_to(self, region):
        """
        Checks if there is a connection from the current region to the specified region.
        
        Args:
            region (str): Name of the region to check for connection.
        
        Returns:
            bool: True if there is a connection, otherwise False.
        """
        return region in self.connections


class mRNN(nn.Module):
    """
    A Multi-Regional Recurrent Neural Network (mRNN) that implements interactions between brain regions.
    This model is designed to simulate neural interactions between different brain areas, with support
    for region-specific properties and inter-regional connections.

    Key Features:
    - Supports multiple brain regions with distinct properties
    - Implements Dale's Law for biological plausibility
    - Handles region-specific cell types
    - Includes noise injection for both hidden states and inputs
    - Supports tonic (baseline) firing rates for each region

    Args:
        config (str): Path to JSON configuration file specifying network architecture
        inp_dim (int): Dimension of the input
        noise_level_act (float, optional): Noise level for activations. Defaults to 0.01
        noise_level_inp (float, optional): Noise level for inputs. Defaults to 0.01
        constrained (bool, optional): Whether to apply Dale's Law constraints. Defaults to True
        t_const (float, optional): Time constant for network dynamics. Defaults to 0.1
        device (str, optional): Computing device to use. Defaults to "cuda"
    """

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
        
        # Initialize network parameters
        self.region_dict = {}
        self.inp_dim = inp_dim
        self.constrained = constrained
        self.device = device
        self.t_const = t_const
        self.sigma_recur = noise_level_act
        self.sigma_input = noise_level_inp

        # Load and process configuration
        with open(config, 'r') as f:
            config = json.load(f)
        
        # Generate network structure
        self.__gen_regions(config["regions"])
        self.__gen_connections(config["connections"])
        
        # Generate weight matrices and masks
        self.W_rec, self.W_rec_mask, self.W_rec_sign_matrix = self.gen_w_rec()
        self.total_num_units = self.__get_total_num_units()
        self.tonic_inp = self.__get_tonic_inp()

        # Get indices for specific regions
        self.alm_start_idx, self.alm_end_idx = self.get_region_indices("alm")
        self.thal_mask = self.__gen_region_mask("thal")

    def gen_w_rec(self):
        """
        Generates the full recurrent connectivity matrix and associated masks.
        
        Returns:
            tuple: (W_rec, W_rec_mask, W_rec_sign_matrix)
                - W_rec: Learnable weight matrix
                - W_rec_mask: Binary mask for allowed connections
                - W_rec_sign_matrix: Sign constraints for Dale's Law
        """
        region_connection_columns = []
        region_weight_mask_columns = []
        region_sign_matrix_columns = []

        for cur_region in self.region_dict:
            self.__get_full_connectivity(self.region_dict[cur_region])

            # Collect connections, masks, and sign matrices for current region
            connections_from_region = []
            weight_mask_from_region = []
            sign_matrix_from_region = []

            for connection in self.region_dict.keys():
                region_data = self.region_dict[cur_region].connections[connection]
                connections_from_region.append(region_data["parameter"])
                weight_mask_from_region.append(region_data["weight_mask"])
                sign_matrix_from_region.append(region_data["sign_matrix"])
            
            # Concatenate region-specific matrices
            region_connection_columns.append(torch.cat(connections_from_region, dim=0))
            region_weight_mask_columns.append(torch.cat(weight_mask_from_region, dim=0))
            region_sign_matrix_columns.append(torch.cat(sign_matrix_from_region, dim=0))
        
        # Create final matrices
        W_rec = nn.Parameter(torch.cat(region_connection_columns, dim=1))
        W_rec_mask = torch.cat(region_weight_mask_columns, dim=1)
        W_rec_sign = torch.cat(region_sign_matrix_columns, dim=1)

        return W_rec, W_rec_mask, W_rec_sign

    def apply_dales_law(self):
        """
        Applies Dale's Law constraints to the recurrent weight matrix.
        Dale's Law states that a neuron can be either excitatory or inhibitory, but not both.
        
        Returns:
            torch.Tensor: Constrained weight matrix
        """
        return (self.W_rec_mask * F.relu(self.W_rec)) * self.W_rec_sign_matrix

    def forward(self, inp, cue_inp, hn, xn, inhib_stim, noise=True):
        """
        Forward pass through the network.

        Args:
            inp (torch.Tensor): Input sequence (target timing)
            cue_inp (torch.Tensor): Cue input sequence
            hn (torch.Tensor): Hidden state
            xn (torch.Tensor): Pre-activation hidden state
            inhib_stim (torch.Tensor): Inhibitory stimulus
            noise (bool, optional): Whether to apply noise. Defaults to True

        Returns:
            torch.Tensor: Network output sequence
        """
        hn_next = hn.squeeze(0)
        xn_next = xn.squeeze(0)
        size = inp.shape[1]
        new_hs = []
        new_xs = []

        # Apply Dale's Law if constrained
        W_rec = self.apply_dales_law() if self.constrained else self.W_rec

        # Calculate noise terms
        if noise:
            perturb_hid = np.sqrt(2 * self.t_const * self.sigma_recur**2) * np.random.normal(0, 1)
            perturb_inp = np.sqrt(2 * self.t_const * self.sigma_input**2) * np.random.normal(0, 1)
        else:
            perturb_hid = perturb_inp = 0

        # Process sequence
        for t in range(size):
            # Prepare ITI input
            iti_act = inp[:, t, :] + perturb_inp
            non_iti_mask = torch.zeros(
                size=(iti_act.shape[0], self.total_num_units - self.region_dict["iti"].num_units),
                device=self.device
            )
            iti_input = torch.cat([non_iti_mask, iti_act], dim=-1)

            # Update hidden state
            xn_next = (xn_next 
                      + self.t_const 
                      * (-xn_next
                         + (W_rec @ hn_next.T).T
                         + iti_input
                         + self.tonic_inp
                         + inhib_stim[:, t, :]
                         + (cue_inp[:, t, :] * self.thal_mask)
                         + perturb_hid))

            hn_next = F.relu(xn_next)
            new_xs.append(xn_next)
            new_hs.append(hn_next)
        
        return torch.stack(new_hs, dim=1)

    def get_region_indices(self, region):
        """
        Gets the start and end indices for a specific region in the hidden state vector.

        Args:
            region (str): Name of the region

        Returns:
            tuple: (start_idx, end_idx)
        """
        start_idx = 0
        for cur_reg in self.region_dict:
            if cur_reg == region:
                return start_idx, start_idx + self.region_dict[cur_reg].num_units
            start_idx += self.region_dict[cur_reg].num_units
        return start_idx, start_idx

    def __gen_regions(self, regions):
        """
        Generates region objects from configuration.

        Args:
            regions (list): List of region configurations
        """
        for region in regions:
            self.region_dict[region["name"]] = Region(
                num_units=region["num_units"],
                base_firing=region["base_firing"],
                device=self.device,
                cell_types=region["cell_types"]
            )

    def __gen_connections(self, connections):
        """
        Generates inter-regional connections from configuration.

        Args:
            connections (list): List of connection configurations
        """
        for connection in connections:
            self.region_dict[connection["src_region"]].add_connection(
                proj_region_name=connection["dst_region"],
                proj_region=self.region_dict[connection["dst_region"]],
                src_region_cell_type=connection["src_region_cell_type"],
                dst_region_cell_type=connection["dst_region_cell_type"],
                sign=connection["sign"]
            )

    def __gen_region_mask(self, region, cell_type=None):
        """
        Generates a mask for a specific region and optionally a cell type.

        Args:
            region (str): Region name
            cell_type (str, optional): Cell type within region. Defaults to None

        Returns:
            torch.Tensor: Binary mask
        """
        mask_type = "full" if cell_type is None else cell_type
        mask = []
        
        for next_region in self.region_dict:
            if region == next_region:
                mask.append(self.region_dict[region].masks[mask_type])
            else:
                mask.append(self.region_dict[next_region].masks["zero"])
        
        return torch.cat(mask).to(self.device)

    def __get_full_connectivity(self, region):
        """
        Ensures all possible connections are defined for a region, adding zero
        connections where none are specified.

        Args:
            region (Region): Region object to complete connections for
        """
        for other_region in self.region_dict:
            if not region.has_connection_to(other_region):
                region.add_connection(
                    other_region,
                    self.region_dict[other_region],
                    src_region_cell_type=None,
                    dst_region_cell_type=None,
                    sign=None,
                    zero_connection=True
                )

    def __get_total_num_units(self):
        """
        Calculates total number of units across all regions.

        Returns:
            int: Total number of units
        """
        return sum(region.num_units for region in self.region_dict.values())

    def __get_tonic_inp(self):
        """
        Collects baseline firing rates for all regions.

        Returns:
            torch.Tensor: Vector of baseline firing rates
        """
        return torch.cat([region.base_firing for region in self.region_dict.values()]).to(self.device) 

class CBGTCL(nn.Module):
    def __init__(
        self,
        config,
        inp_dim,
        out_dim,
        out_type,
        noise_level_act=0.1,
        noise_level_inp=0.1,
        constrained=True,
        device="cuda"
    ):
        super(CBGTCL, self).__init__()

        self.inp_dim = inp_dim
        self.out_dim = out_dim
        self.out_type = out_type
        self.constrained = constrained
        self.device = device

        self.mrnn = mRNN(
            config,
            inp_dim,
            noise_level_act=noise_level_act,
            noise_level_inp=noise_level_inp,
            constrained=constrained 
        )

        self.total_num_units = self.mrnn.total_num_units

        # Output weights
        if out_type == "ramp":

            self.out_weight_alm = (1 / self.mrnn.region_dict["alm"].num_units) * torch.ones(size=(out_dim, self.mrnn.region_dict["alm"].num_units)).to(self.device)

        elif out_type == "data":

            self.out_weight_alm = nn.Parameter(torch.empty(size=(out_dim, self.mrnn.region_dict["alm"].num_units))).to(self.device)
            nn.init.uniform_(self.out_weight_alm, 0, 1)
    
    def forward(
        self,
        inp, 
        cue_inp, 
        hn, 
        xn, 
        inhib_stim, 
        noise=True
    ):

        outs = []
        
        hn = self.mrnn(
            inp, 
            cue_inp, 
            hn, 
            xn, 
            inhib_stim, 
            noise
        )

        alm_start, alm_end = self.mrnn.get_region_indices("alm")

        for t in range(hn.shape[1]):
            outs.append((self.out_weight_alm @ hn[:, t, alm_start:alm_end].T).T)
        
        out = torch.stack(outs, dim=1)

        return hn, out
        