import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sac_model import weights_init_
from torch.distributions import Normal

class ThalamoCortical(nn.Module):
    def __init__(self, inp_dim, hid):
        super(ThalamoCortical, self).__init__()
        self.inp_dim = inp_dim
        self.hid = hid

        # Cortical Weights
        self.J_cc = torch.randn(hid, hid) / np.sqrt(hid)
        self.J_ct = torch.randn(hid, inp_dim) / np.sqrt(hid)

        # Thalamic Weights
        self.J_tc = torch.randn(inp_dim, hid) / np.sqrt(hid)

        # Thalamic Timescale (not sure what to put)
        self.tau = 1.

        # Readout for probability
        self.W_out = torch.randn(hid,) / np.sqrt(hid)

        self.cortical_activity = torch.zeros(size=(hid,))
        self.thalamic_activity = torch.zeros(size=(inp_dim,))

    # TODO initialize weights of thalamocortical-corticothalamal network 
    # such that a specific selection from BG results in a lick at a certain timestep (if possible)
    # BG will learn to generate sustained activity specific to a particular motif (lick at 1s or 3s)
    def forward(self, x):

        # discrete dynamics with forward euler (dt = 1)
        self.thalamic_activity = self.thalamic_activity - (1/self.tau) * self.thalamic_activity + self.J_tc @ self.cortical_activity + x
        self.cortical_activity = self.J_cc @ self.cortical_activity + self.J_ct @ F.relu(self.thalamic_activity)
        lick_prob = F.sigmoid(self.W_out @ self.cortical_activity)

        return lick_prob
