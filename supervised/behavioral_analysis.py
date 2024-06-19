import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional, RNN_MultiRegional_NoConstraint, RNN_MultiRegional_NoConstraintThal
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_delay_data, get_acts, get_ramp

CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDS = 3
MODEL_TYPE = "constraint"
CONDITION = 0

def get_lick_samples(rnn, x_data, model_type, num_samples=500):

    decision_times = []
    for sample in range(num_samples):

        if model_type == "constraint":
            hn = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
        elif model_type == "no_constraint":
            hn = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
        elif model_type == "no_constraint_thal":
            hn = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()

        out, _, _, _, _ = rnn(x_data, hn, x, 0, noise=True)

        for i, logit in enumerate(out[CONDITION, 1000:, :]):
            num = np.random.uniform(0, 1)
            if num < logit:
                decision_times.append(i * DT)
                break
    
    return decision_times

def calculate_ecdf(decision_times):
    
    bins = np.linspace(0, 2.1, 1000)
    bin_probs = []
    for bin in bins:
        prob = 0
        for time in decision_times:
            if time < bin:
                prob = prob + 1
        prob = prob / len(decision_times)
        bin_probs.append(prob)
    return bin_probs

def plot_ecdf(bin_probs):
    
    plt.plot(bin_probs)
    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "constraint":
        rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "no_constraint":
        rnn = RNN_MultiRegional_NoConstraint(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "no_constraint_thal":
        rnn = RNN_MultiRegional_NoConstraintThal(INP_DIM, HID_DIM, OUT_DIM).cuda()

    rnn.load_state_dict(checkpoint)

    x_data, _, len_seq = gather_delay_data(dt=0.001, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    decision_times = get_lick_samples(rnn, x_data, MODEL_TYPE)
    bin_probs = calculate_ecdf(decision_times)
    plot_ecdf(bin_probs)
    
if __name__ == "__main__":
    main()