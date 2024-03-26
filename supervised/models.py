import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

# Actor RNN
class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, sparse=False):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.rnn = nn.GRU(inp_dim, hid_dim, batch_first=True, num_layers=1)

        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, len_seq=None):

        x = pack_padded_sequence(x, len_seq,  batch_first=True, enforce_sorted=False)

        rnn_x, hn = self.rnn(x, hn)

        rnn_x, _ = pad_packed_sequence(rnn_x, batch_first=True)

        out = self.fc2(rnn_x)
        
        return out, hn, rnn_x

class RNN_Delay(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, sparse=False):
        super(RNN_Delay, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.rnn = nn.RNN(inp_dim, hid_dim, batch_first=True, num_layers=1, nonlinearity="relu")
        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, len_seq=None):

        x = pack_padded_sequence(x, len_seq,  batch_first=True, enforce_sorted=False)

        rnn_x, hn = self.rnn(x, hn)

        rnn_x, _ = pad_packed_sequence(rnn_x, batch_first=True)

        out = torch.sigmoid(self.fc2(rnn_x))
        
        return out, hn, rnn_x