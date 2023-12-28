import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.distributions import Categorical

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
    def __init__(self, inp_dim, hid_dim, action_dim):
        super(Actor, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True, num_layers=1)
        self.fc2 = nn.Linear(hid_dim, action_dim)

        
    def forward(self, x: torch.Tensor, hn: torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        hn = hn.cuda()

        x = F.relu(self.fc1(x))

        gru_x, hn = self.gru(x, hn)

        x = F.softmax(self.fc2(gru_x), dim=-1)

        m = Categorical(x)

        action = m.sample()

        log_prob = m.log_prob(action)
        
        return action, log_prob, hn, gru_x
    
# Critic RNN
class Critic(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super(Critic, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.gru = nn.GRU(hid_dim, hid_dim, batch_first=True, num_layers=1)
        self.fc2 = nn.Linear(hid_dim, 1)

    def forward(self, x: torch.Tensor, hn: torch.Tensor) -> (int, int):

        hn = hn.cuda()

        x = F.relu(self.fc1(x))
        x, hn = self.gru(x, hn)
        x = self.fc2(x)

        return x