import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_normal_(m.weight, gain=.5)
        torch.nn.init.constant_(m.bias, 0)

# Actor RNN
class Actor(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, action_scale, action_bias):
        super(Actor, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.weight_l0_ih = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        self.bias_l0_hh = nn.Parameter(torch.empty(size=(hid_dim,)))
        self.bias_l0_ih = nn.Parameter(torch.empty(size=(hid_dim,)))
        nn.init.xavier_uniform_(self.weight_l0_hh)
        nn.init.xavier_uniform_(self.weight_l0_ih)
        nn.init.uniform_(self.bias_l0_hh)
        nn.init.uniform_(self.bias_l0_ih)

        self.mean_linear = nn.Linear(hid_dim, action_dim)
        self.std_linear = nn.Linear(hid_dim, action_dim)

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x: torch.Tensor, hn: torch.Tensor, sampling=True, len_seq=None):
        
        hn_next = hn.squeeze(0)
        new_hs = []
        for t in range(x.shape[1]):
            hn_next = torch.sigmoid(hn_next @ self.weight_l0_hh + x[:, t, :] @ self.weight_l0_ih + self.bias_l0_hh + self.bias_l0_ih)
            new_hs.append(hn_next)
        rnn_out = torch.stack(new_hs, dim=1)

        mean = self.mean_linear(rnn_out)
        std = self.std_linear(rnn_out)
        std = torch.clamp(std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, std, rnn_out, hn
    
    def sample(self, state: torch.Tensor, hn: torch.Tensor, sampling: bool=True, len_seq: list=None):

        hn = hn.cuda()
        
        mean, log_std, rnn_out, hn = self.forward(state, hn, sampling, len_seq)

        mean_size = mean.size()
        log_std_size = log_std.size()

        mean = mean.reshape(-1, mean.size()[-1])
        log_std = log_std.reshape(-1, log_std.size()[-1])

        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)

        # Enforce the action_bounds
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        if sampling == False:
            action = action.reshape(mean_size[0], mean_size[1], mean_size[2])
            log_prob = log_prob.reshape(log_std_size[0], log_std_size[1], 1) 
            mean = mean.reshape(mean_size[0], mean_size[1], mean_size[2])

        return action, log_prob, mean, rnn_out, hn


# Critic RNN
class Critic(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super(Critic, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        
        # First RNN
        self.fc11 = nn.Linear(inp_dim, hid_dim)

        self.weight_l0_hh_1 = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.weight_l0_ih_1 = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.bias_l0_hh_1 = nn.Parameter(torch.empty(size=(hid_dim,)))
        self.bias_l0_ih_1 = nn.Parameter(torch.empty(size=(hid_dim,)))
        nn.init.xavier_uniform_(self.weight_l0_hh_1)
        nn.init.xavier_uniform_(self.weight_l0_ih_1)
        nn.init.uniform_(self.bias_l0_hh_1)
        nn.init.uniform_(self.bias_l0_ih_1)

        self.fc12 = nn.Linear(hid_dim, 1)

        # Second RNN
        self.fc21 = nn.Linear(inp_dim, hid_dim)

        self.weight_l0_hh_2 = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.weight_l0_ih_2 = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.bias_l0_hh_2 = nn.Parameter(torch.empty(size=(hid_dim,)))
        self.bias_l0_ih_2 = nn.Parameter(torch.empty(size=(hid_dim,)))
        nn.init.xavier_uniform_(self.weight_l0_hh_2)
        nn.init.xavier_uniform_(self.weight_l0_ih_2)
        nn.init.uniform_(self.bias_l0_hh_2)
        nn.init.uniform_(self.bias_l0_ih_2)

        self.fc22 = nn.Linear(hid_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, hn: torch.Tensor, len_seq: list=None):

        x = torch.cat((state, action), dim=-1)
        hn = hn.cuda()

        x1 = F.relu(self.fc11(x))

        hn_next = hn.squeeze(0)
        new_hs = []
        for t in range(x1.shape[1]):
            hn_next = torch.sigmoid(hn_next @ self.weight_l0_hh_1 + x1[:, t, :] @ self.weight_l0_ih_1 + self.bias_l0_hh_1 + self.bias_l0_ih_1)
            new_hs.append(hn_next)
        rnn_out_1 = torch.stack(new_hs, dim=1)

        x1 = self.fc12(rnn_out_1)

        x2 = F.relu(self.fc21(x))

        hn_next = hn.squeeze(0)
        new_hs = []
        for t in range(x2.shape[1]):
            hn_next = torch.sigmoid(hn_next @ self.weight_l0_hh_2 + x2[:, t, :] @ self.weight_l0_ih_2 + self.bias_l0_hh_2 + self.bias_l0_ih_2)
            new_hs.append(hn_next)
        rnn_out_2 = torch.stack(new_hs, dim=1)

        x2 = self.fc22(rnn_out_2)

        return x1, x2