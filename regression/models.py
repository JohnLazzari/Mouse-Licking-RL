import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np


class RNN_Seq(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, inhib_upper_bound=-1e-3, excite_lower_bound=1e-3, beta=0.25):
        super(RNN_Seq, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.inhib_upper_bound = inhib_upper_bound
        self.excite_lower_bound = excite_lower_bound
        self.beta = beta
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.weight_ih_l0 = nn.Parameter(torch.zeros(size=(hid_dim, hid_dim)))
        self.weight_hh_l0 = nn.Parameter(torch.zeros(size=(hid_dim, hid_dim)))
        nn.init.uniform_(self.weight_hh_l0, -0.1, inhib_upper_bound)
        nn.init.uniform_(self.weight_ih_l0, excite_lower_bound, 0.1)
        with torch.no_grad():
            eye = torch.eye(hid_dim)
            ones = torch.ones(size=(hid_dim, hid_dim))
            mask = ones - eye
            self.weight_hh_l0 *= mask
        
        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, y_depression: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        x = F.sigmoid(self.fc1(x))
        new_h = []
        for t in range(x.shape[1]):
            new_h.append(F.sigmoid((y_depression * hn) @ self.weight_hh_l0 + x[:, t, :] @ self.weight_ih_l0))
            y_depression = y_depression + (1 / 5) * ( -(y_depression - 1) * (1 - new_h[-1]) - (y_depression - 0.25) * new_h[-1])
            hn = new_h[-1]
        new_h = torch.stack(new_h, dim=1)

        out = self.fc2(new_h)
        
        return out, hn, new_h, y_depression


class RNN_Inhibitory(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, inhib_upper_bound=-1e-3, excite_lower_bound=1e-3, beta=0.25):
        super(RNN_Inhibitory, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.inhib_upper_bound = inhib_upper_bound
        self.excite_lower_bound = excite_lower_bound
        self.beta = beta
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.weight_ih_l0 = nn.Parameter(torch.zeros(size=(hid_dim, hid_dim)))
        self.weight_hh_l0 = nn.Parameter(torch.zeros(size=(hid_dim, hid_dim)))
        nn.init.uniform_(self.weight_hh_l0, -0.1, inhib_upper_bound)
        nn.init.uniform_(self.weight_ih_l0, excite_lower_bound, 0.1)
        with torch.no_grad():
            eye = torch.eye(hid_dim)
            ones = torch.ones(size=(hid_dim, hid_dim))
            mask = ones - eye
            self.weight_hh_l0 *= mask
        
        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, y_depression: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        x = F.sigmoid(self.fc1(x))
        new_h = []
        for t in range(x.shape[1]):
            new_h.append(F.sigmoid(hn @ self.weight_hh_l0 + x[:, t, :] @ self.weight_ih_l0))
            hn = new_h[-1]
        new_h = torch.stack(new_h, dim=1)

        out = self.fc2(new_h)
        
        return out, hn, new_h, y_depression


# Actor RNN
class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.rnn = nn.RNN(hid_dim, hid_dim, batch_first=True, num_layers=1, nonlinearity="relu")
        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, len_seq=None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        x = F.relu(self.fc1(x))

        x = pack_padded_sequence(x, len_seq,  batch_first=True, enforce_sorted=False)

        rnn_x, hn = self.rnn(x, hn)

        rnn_x, _ = pad_packed_sequence(rnn_x, batch_first=True)

        out = self.fc2(rnn_x)
        
        return out, hn, rnn_x