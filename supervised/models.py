import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.weight_l0_ih = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        nn.init.xavier_normal_(self.weight_l0_hh)
        nn.init.xavier_normal_(self.weight_l0_ih)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, inp: torch.Tensor, hn: torch.Tensor):

        hn_next = hn.squeeze(0)
        new_hs = []
        for t in range(inp.shape[1]):
            hn_next = torch.sigmoid(hn_next @ self.weight_l0_hh + inp[:, t, :] @ self.weight_l0_ih)
            new_hs.append(hn_next)
        rnn_out = torch.stack(new_hs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)

        out = F.relu(self.fc1(rnn_out))
        out = torch.sigmoid(self.fc2(out))
        
        return out, hn_last, rnn_out