import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
import math

def sparse_init(
    tensor,
    sparsity,
    std=0.01
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
    num_zeros = int(math.ceil(sparsity * rows))

    with torch.no_grad():
        tensor.uniform_(-.1, 0)
        for col_idx in range(cols):
            row_indices = torch.randperm(rows)
            zero_indices = row_indices[:num_zeros]
            tensor[zero_indices, col_idx] = 0
    return tensor

# Actor RNN
class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, sparse=False):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.fc1 = nn.Linear(inp_dim, hid_dim)
        self.rnn = nn.GRU(hid_dim, hid_dim, batch_first=True, num_layers=1)

        if sparse:
            sparse_init(self.rnn.weight_hh_l0, 0.5)
            nn.init.zeros_(self.rnn.bias_hh_l0)
            self.rnn.weight_hh_l0.requires_grad = False
            self.rnn.bias_hh_l0.requires_grad = False

        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, x: torch.Tensor, hn: torch.Tensor, len_seq=None):

        x = F.relu(self.fc1(x))

        x = pack_padded_sequence(x, len_seq,  batch_first=True, enforce_sorted=False)

        rnn_x, hn = self.rnn(x, hn)

        rnn_x, _ = pad_packed_sequence(rnn_x, batch_first=True)

        out = self.fc2(rnn_x)
        
        return out, hn, rnn_x