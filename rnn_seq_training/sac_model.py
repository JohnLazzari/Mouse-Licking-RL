import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-6

class Actor_Seq(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, inhib_upper_bound=-1e-3, excite_lower_bound=1e-3, beta=0.25):
        super(Actor_Seq, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.inhib_upper_bound = inhib_upper_bound
        self.excite_lower_bound = excite_lower_bound
        self.beta = beta
        
        self.weight_ih_l0 = nn.Parameter(torch.zeros(size=(inp_dim, hid_dim)))
        self.weight_hh_l0 = nn.Parameter(torch.zeros(size=(hid_dim, hid_dim)))
        nn.init.uniform_(self.weight_hh_l0, -np.sqrt(6 / (hid_dim + hid_dim)), inhib_upper_bound)
        nn.init.uniform_(self.weight_ih_l0, excite_lower_bound, np.sqrt(6 / (hid_dim + hid_dim)))
        
        self.mean_linear = nn.Linear(hid_dim, action_dim)
        #nn.init.uniform_(self.mean_linear.weight, -np.sqrt(6 / (hid_dim + hid_dim)), inhib_upper_bound)
        #nn.init.zeros_(self.mean_linear.bias)

        self.std_linear = nn.Linear(hid_dim, action_dim)

        # Range of actions from -1 to 1
        self.action_scale = .5
        self.action_bias = .5

    def forward(self, x: torch.Tensor, hn: torch.Tensor, y_depression: torch.Tensor, y_ones: torch.Tensor, y_beta: torch.Tensor, sampling=True, len_seq=None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        new_x = []
        for t in range(x.shape[1]):
            new_x.append(self.gain_sigmoid((y_depression * hn) @ self.weight_hh_l0 + x[:, t, :] @ self.weight_ih_l0))
            y_depression = y_depression - (1 / 20) * ( (y_depression - y_ones) * (y_ones - new_x[-1]) - (y_depression - y_beta) * new_x[-1])
        hn = new_x[-1]
        new_x = torch.stack(new_x, dim=1)

        mean = self.mean_linear(new_x)
        std = self.std_linear(new_x)
        std = torch.clamp(std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, std, hn, new_x, y_depression

    def gain_sigmoid(self, x, gain=20):
        return 1 / (1 + torch.exp(-gain * x))
    
    def sample(self, state: torch.Tensor, hn: torch.Tensor, y_depression: torch.Tensor, y_ones: torch.Tensor, y_beta: torch.Tensor, sampling: bool = True, len_seq: list = None) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        hn = hn.cuda()
        
        mean, log_std, h_current, out, y_depression = self.forward(state, hn, y_depression, y_ones, y_beta, sampling, len_seq)
        #if sampling == False; then reshape mean and log_std from (B, L_max, A) to (B*Lmax, A)

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

        return action, log_prob, mean, h_current, y_depression, out

# Critic RNN
class Critic(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super(Critic, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        
        self.gru1 = nn.GRU(inp_dim, hid_dim, batch_first=True, num_layers=1)
        self.fc12 = nn.Linear(hid_dim, 1)

        self.gru2 = nn.GRU(inp_dim, hid_dim, batch_first=True, num_layers=1)
        self.fc22 = nn.Linear(hid_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, hn: torch.Tensor, len_seq: bool = None) -> (int, int):

        x = torch.cat((state, action), dim=-1)
        hn = hn.cuda()

        x1 = pack_padded_sequence(x, len_seq, batch_first=True, enforce_sorted=False)
        x1, hn1 = self.gru1(x1, hn)
        x1, _ = pad_packed_sequence(x1, batch_first=True)
        x1 = self.fc12(x1)

        x2 = pack_padded_sequence(x, len_seq, batch_first=True, enforce_sorted=False)
        x2, hn2 = self.gru2(x2, hn)
        x2, _ = pad_packed_sequence(x2, batch_first=True)
        x2 = self.fc22(x2)

        return x1, x2