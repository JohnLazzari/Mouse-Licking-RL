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

def sparse_(
    tensor,
):
    r"""Fill the 2D input `Tensor` as a sparse matrix.

    The non-zero elements will be drawn from the normal distribution
    :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
    Hessian-free optimization` - Martens, J. (2010).

    Args:
        tensor: an n-dimensional `torch.Tensor`
        sparsity: The fraction of elements in each column to be set to zero
        std: the standard deviation of the normal distribution used to generate
            the non-zero values
        generator: the torch Generator to sample from (default: None)

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.sparse_(w, sparsity=0.1)
    """
    if tensor.ndimension() != 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    rows, cols = tensor.shape
    sparsity = np.random.uniform(0.5, 0.9, size=(cols,))
    num_zeros = np.ceil(sparsity * rows).astype(int)

    with torch.no_grad():
        tensor.uniform_(-.25, 0)
        for col_idx, col_zeros in enumerate(num_zeros):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:col_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor

# Actor RNN
class Actor(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, action_scale, action_bias):
        super(Actor, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.beta = 0.25
        
        self.weight_hh_l0 = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.weight_ih_l0 = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        # Add asynchrony in initialization
        sparse_(self.weight_hh_l0)
        inv_eye = -torch.eye(hid_dim, hid_dim) + 1
        nn.init.xavier_uniform_(self.weight_ih_l0)
        self.weight_hh_l0.requires_grad = False
        self.weight_hh_l0 *= inv_eye
        
        self.mean_linear = nn.Linear(hid_dim, action_dim)
        self.std_linear = nn.Linear(hid_dim, action_dim)

        self.action_scale = action_scale
        self.action_bias = action_bias

    def forward(self, x: torch.Tensor, hn: torch.Tensor, y_depression: torch.Tensor, sampling=True, len_seq=None):

        new_hs = []
        # Assuming batch first is True
        h_cur = hn
        for step in range(x.shape[1]):
            h_cur = torch.sigmoid((y_depression.squeeze(0) * h_cur.squeeze(0)) @ self.weight_hh_l0 + x[:, step, :] @ self.weight_ih_l0)
            y_depression = y_depression + (1/10) * (-(y_depression - 1) * (1 - h_cur) - (y_depression - self.beta) * h_cur)
            new_hs.append(h_cur)
        h_last = h_cur.unsqueeze(0)
        all_hs = torch.stack(new_hs, dim=1)

        mean = self.mean_linear(all_hs)
        std = self.std_linear(all_hs)
        std = torch.clamp(std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        
        return mean, std, h_last, all_hs, y_depression.unsqueeze(0)
    
    def sample(self, state: torch.Tensor, hn: torch.Tensor, y_depression, sampling: bool = True, len_seq: list = None):

        hn = hn.cuda()
        
        mean, log_std, h_current, gru_out, y_depression = self.forward(state, hn, y_depression, sampling, len_seq)
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

        return action, log_prob, mean, h_current, gru_out, y_depression


# Critic RNN
class Critic(nn.Module):
    def __init__(self, inp_dim: int, hid_dim: int):
        super(Critic, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        
        self.fc11 = nn.Linear(inp_dim+hid_dim, hid_dim)
        #self.gru1 = nn.GRU(hid_dim, hid_dim, batch_first=True, num_layers=1)
        self.fc12 = nn.Linear(hid_dim, 1)

        self.fc21 = nn.Linear(inp_dim+hid_dim, hid_dim)
        #self.gru2 = nn.GRU(hid_dim, hid_dim, batch_first=True, num_layers=1)
        self.fc22 = nn.Linear(hid_dim, 1)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor, hn: torch.Tensor, len_seq: bool = None):

        x = torch.cat((state, hn, action), dim=-1)
        hn = hn.cuda()

        x1 = F.relu(self.fc11(x))
        #x1 = pack_padded_sequence(x1, len_seq, batch_first=True, enforce_sorted=False)
        #x1, hn1 = self.gru1(x1, hn)
        #x1, _ = pad_packed_sequence(x1, batch_first=True)
        x1 = self.fc12(x1)

        x2 = F.relu(self.fc21(x))
        #x2 = pack_padded_sequence(x2, len_seq, batch_first=True, enforce_sorted=False)
        #x2, hn2 = self.gru2(x2, hn)
        #x2, _ = pad_packed_sequence(x2, batch_first=True)
        x2 = self.fc22(x2)

        return x1, x2