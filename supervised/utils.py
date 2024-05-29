import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def NormalizeData(data, min, max):
    '''
        Min-Max normalization for any data

        data: full array of data
        min: minimum of data along each row
        max: maximum of data along each row
    '''
    return (data - min) / (max - min)

def gather_delay_data(dt, hid_dim):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''
    
    inp = {}
    lick_target = {}

    # Condition 1: 0.3 for 1.1s
    inp[0] = 0.9*torch.ones(size=(int(1.1 / dt), 1))

    # Condition 2: 0.6 for 1.6
    inp[1] = 0.6*torch.ones(size=(int(1.6 / dt), 1))

    # Condition 3: 0.9 for 2.1s
    inp[2] = 0.3*torch.ones(size=(int(2.1 / dt), 1))

    # Combine all inputs
    total_inp = pad_sequence([inp[0], inp[1], inp[2]], batch_first=True)

    # Lick range is 0.1s
    lick = torch.ones(size=(int(0.1 / dt), 1))

    # Condition 1: 1-1.1 lick
    no_lick = torch.zeros(size=(int(1 / dt), 1))
    total_lick = torch.cat([no_lick, lick], dim=0)
    lick_target[0] = total_lick

    # Condition 2: 1.5-1.6 lick
    no_lick = torch.zeros(size=(int(1.5 / dt), 1))
    total_lick = torch.cat([no_lick, lick], dim=0)
    lick_target[1] = total_lick

    # Condition 3: 2-2.1 lick
    no_lick = torch.zeros(size=(int(2 / dt), 1))
    total_lick = torch.cat([no_lick, lick], dim=0)
    lick_target[2] = total_lick

    # Combine all targets
    total_target = pad_sequence([lick_target[0], lick_target[1], lick_target[2]], batch_first=True)

    # Combine all sequence lengths
    len_seq = [int(1.1 / dt), int(1.6 / dt), int(2.1 / dt)]

    return total_inp, total_target, len_seq

def get_ramp(dt):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    all_ramps = {}
    for cond in range(3):
        all_ramps[cond] = torch.linspace(0, 1, int((1.1 + 0.5 * cond) / dt), dtype=torch.float32).unsqueeze(1)
    total_ramp = pad_sequence([all_ramps[0], all_ramps[1], all_ramps[2]], batch_first=True)
    return total_ramp

def get_acts(len_seq, rnn, hid_dim, x_data, cond, perturbation, perturbation_strength=None, region="None"):

    '''
        If silencing multi-regional model, get the activations with and without silencing
    '''

    acts = []
    hn = torch.zeros(size=(1, 1, hid_dim)).cuda()
    x = torch.zeros(size=(1, 1, hid_dim)).cuda()

    if region == "alm":
        region_mask = torch.cat([torch.ones(size=(int(hid_dim/2),)), 
                                 perturbation_strength * torch.ones(size=(int(hid_dim/2),))]).unsqueeze(0).unsqueeze(0).cuda()
    elif region == "str":
        region_mask = torch.cat([perturbation_strength * torch.ones(size=(int(hid_dim/2),)), 
                                torch.ones(size=(int(hid_dim/2),))]).unsqueeze(0).unsqueeze(0).cuda()

    for t in range(len_seq):
        with torch.no_grad():        
            _, hn, _, x, _ = rnn(x_data[cond:cond+1, t:t+1, :], hn, x)
            if perturbation == True and t > 500 and t < 800:
                hn = hn * region_mask
                x = x * region_mask
            acts.append(hn.squeeze().cpu().numpy())
    
    return np.array(acts)