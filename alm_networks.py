import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sac_model import weights_init_
from torch.distributions import Normal

class ALM(nn.Module):
    def __init__(self, action_dim, alm_hid):
        super(ALM, self).__init__()
        self.action_dim = action_dim
        self.alm_hid = alm_hid
        self._alm = nn.RNN(action_dim, alm_hid, batch_first=True, nonlinearity='tanh')
        self._alm_out = nn.Linear(alm_hid, 3)

    def forward(self, x, hn):

        action, hn = self._alm(x, hn)
        action = F.sigmoid(self._alm_out(action))

        return action, hn


class ALM_Values(nn.Module):
    def __init__(self, alm_hid):
        super(ALM_Values, self).__init__()

        self.linear1 = nn.Linear(3, alm_hid)
        self.linear2 = nn.Linear(alm_hid, alm_hid)
        self.linear3 = nn.Linear(alm_hid, 1)
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x