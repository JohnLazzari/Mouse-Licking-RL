import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import rankdata, spearmanr

def NormalizeData(data, min, max):
    '''
        Min-Max normalization for any data

        data: full array of data
        min: minimum of data along each row
        max: maximum of data along each row
    '''
    return (data - min) / (max - min)

def gaussian_density(x, mean, std):
    return torch.exp(-(x - mean)**2/(2*std**2))

def gather_inp_data(dt, hid_dim, ramp):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''
    
    inp = {}

    # Condition 1: 1.1s
    inp[0] = F.relu(ramp[0, 1:, :] - ramp[0, :-1, :]).repeat(1, int(hid_dim * 0.1)).cpu()

    # Condition 2: 1.4s
    inp[1] = F.relu(ramp[1, 1:, :] - ramp[1, :-1, :]).repeat(1, int(hid_dim * 0.1)).cpu()

    # Condition 3: 1.7s
    inp[2] = F.relu(ramp[2, 1:, :] - ramp[2, :-1, :]).repeat(1, int(hid_dim * 0.1)).cpu()

    # Condition 4: 2s
    inp[3] = F.relu(ramp[3, 1:, :] - ramp[3, :-1, :]).repeat(1, int(hid_dim * 0.1)).cpu()

    # Combine all inputs
    total_iti_inp = pad_sequence([inp[0], inp[1], inp[2], inp[3]], batch_first=True)

    zero = torch.zeros(size=(total_iti_inp.shape[0], 1, total_iti_inp.shape[2]))

    total_iti_inp = 10 * torch.cat([
        zero,
        total_iti_inp
    ], dim=1)

    plt.plot(np.mean(total_iti_inp.numpy(), axis=-1).T)
    plt.show()

    # Cue Input
    cue_inp_dict = {}

    for cond in range(4):

        cue_inp_dict[cond] = torch.zeros(size=(int((2.1 + 0.3 * cond) / dt), 1))
        #cue_inp_dict[cond][999:999+100] = 0.01

    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3]], batch_first=True)

    # Combine all sequence lengths
    len_seq = [int(2.1 / dt), int(2.4 / dt), int(2.7 / dt), int(3 / dt)]

    total_inp = [total_iti_inp, total_cue_inp]

    return total_inp, len_seq

def get_ramp(dt):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    
    ramps = {}

    means = [1.1, 1.4, 1.7, 2.0]
    std = [0.4, 0.5, 0.6, 0.7]

    for cond in range(4):

        timepoints = torch.linspace(-1, 2, steps=300).unsqueeze(-1)
        ramps[cond] = gaussian_density(timepoints, means[cond], std[cond])

    total_ramp = pad_sequence([ramps[0], ramps[1], ramps[2], ramps[3]], batch_first=True)

    plt.plot(total_ramp.squeeze().numpy().T)
    plt.show()

    return total_ramp

def get_acts_control(len_seq, rnn, hid_dim, inp_dim, x_data, model_type):

    '''
        Get the activities of the desired region for a single condition (silencing or activation)

        Params:
            len_seq:            list of all sequence lengths for each condition
            rnn:                the RNN to get activities from
            hid_dim:            number of neurons in a single region
            x_data:             inp data (list: iti_inp, cue_inp)
            cond:               condition to analyze (0, 1, 2)
            model_type:         which pathway model to study, chooses hidden state based on this
            ITI_steps:          number of ITI_steps
            start_silence:      number of steps (total, starting from beginning of trial) in which to start silencing
            end_silence:        number of steps in which to end silencing
            stim_strength:      scale of silencing or activation
            extra_steps:        number of extra steps after silencing to look at
            region:             region being silenced or activated
            total_num_units:    total number of units (hid_dim * num_regions)
    '''

    if model_type == "d1d2":
        hn = torch.zeros(size=(1, 4, hid_dim * 6 + inp_dim + int(hid_dim * 0.3))).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 6 + inp_dim + int(hid_dim * 0.3))).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()
    
    inhib_stim = torch.zeros(size=(4, max(len_seq), hn.shape[-1]), device="cuda")

    iti_inp, cue_inp = x_data 
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    with torch.no_grad():        

        _, _, acts = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_acts_manipulation(len_seq, rnn, hid_dim, inp_dim, model_type, start_silence, end_silence, stim_strength, extra_steps, region, dt):

    '''
        Get the activities of the desired region during manipulation for a single condition (silencing or activation)

        Params:
            len_seq:                list of all sequence lengths for each condition
            rnn:                    the RNN to get activities from
            hid_dim:                number of neurons in a single region
            x_data:                 inp data
            cond:                   condition to analyze (0, 1, 2)
            model_type:             which pathway model to study, chooses hidden state based on this
            ITI_steps:              number of ITI_steps
            start_silence:          number of steps (total, starting from beginning of trial) in which to start silencing
            end_silence:            number of steps in which to end silencing
            stim_strength:          scale of silencing or activation
            extra_steps:            number of extra steps after silencing to look at
            region:                 region being silenced or activated
            total_num_units:        total number of units (hid_dim * num_regions)
    '''

    if model_type == "d1d2":
        hn = torch.zeros(size=(1, 4, hid_dim * 6 + inp_dim + int(hid_dim * 0.3))).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 6 + inp_dim + int(hid_dim * 0.3))).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()

    inhib_stim = get_inhib_stim_silence(
        rnn, 
        region, 
        start_silence, 
        end_silence, 
        len_seq, 
        extra_steps, 
        stim_strength, 
        hn.shape[-1]
    )

    iti_inp_silence, cue_inp_silence = get_input_silence(
        dt, 
        hid_dim,
        extra_steps
    )

    iti_inp_silence, cue_inp_silence = iti_inp_silence.cuda(), cue_inp_silence.cuda()
        
    with torch.no_grad():        

        _, _, acts = rnn(iti_inp_silence, cue_inp_silence, hn, xn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_masks(hid_dim, inp_dim, len_seq, regions):

    # mask the losses which correspond to padded values (just in case)
    loss_mask_act = [torch.ones(size=(length, hid_dim * regions + inp_dim + int(hid_dim * 0.3)), dtype=torch.int) for length in len_seq]
    loss_mask_act = pad_sequence(loss_mask_act, batch_first=True).cuda()

    return loss_mask_act

def get_ramp_mode(baseline, peak):
    
    # baseline and peak should be of shape [num_ramps, neurons]
    diff_vec = np.mean(peak - baseline, axis=0)
    diff_vec = diff_vec / np.linalg.norm(diff_vec)
    return diff_vec

def project_ramp_mode(samples, ramp_mode):
    
    # samples should be [time, neurons], ramp_mode should be [neurons]
    projected = samples @ ramp_mode
    return projected

def get_inhib_stim_silence(rnn, region, start_silence, end_silence, len_seq, extra_steps, stim_strength, total_num_units):

    # Select mask based on region being silenced
    if region == "alm":
        mask_inhib_units = stim_strength * (rnn.alm_inhib_mask)
        mask_iti_units = 0 * rnn.iti_mask
        mask = mask_inhib_units + mask_iti_units
    elif region == "str":
        mask = stim_strength * rnn.str_d1_mask
    elif region == "str_d2":
        mask = stim_strength * rnn.str_d2_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # Does this for a single condition, len_seq should be a single number for the chosen condition, and x_data should be [1, len_seq, :]
    inhib_stim_pre = torch.zeros(size=(4, start_silence, total_num_units), device="cuda")
    inhib_stim_silence = torch.ones(size=(4, end_silence - start_silence, total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(4, (max(len_seq) - end_silence) + extra_steps, total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim

def get_input_silence(dt, hid_dim, extra_steps):

    neural_act = get_ramp(dt=dt)
    neural_act = neural_act.cuda()
    x_data, len_seq = gather_inp_data(dt=dt, hid_dim=hid_dim, ramp=neural_act)

    iti_inp_silence, _ = x_data

    iti_inp_silence_cond_1 = torch.cat([
        iti_inp_silence[0, :160, :],
        torch.zeros(size=(60, iti_inp_silence.shape[-1])),
        iti_inp_silence[0, 160:161, :].repeat(10, 1),
        iti_inp_silence[0, 160:, :]
    ])

    iti_inp_silence_cond_2 = torch.cat([
        iti_inp_silence[1, :160, :],
        torch.zeros(size=(60, iti_inp_silence.shape[-1])),
        iti_inp_silence[1, 160:161, :].repeat(10, 1),
        iti_inp_silence[1, 160:, :]
    ])

    iti_inp_silence_cond_3 = torch.cat([
        iti_inp_silence[2, :160, :],
        torch.zeros(size=(60, iti_inp_silence.shape[-1])),
        iti_inp_silence[2, 160:161, :].repeat(10, 1),
        iti_inp_silence[2, 160:, :]
    ])

    iti_inp_silence_cond_4 = torch.cat([
        iti_inp_silence[3, :160, :],
        torch.zeros(size=(60, iti_inp_silence.shape[-1])),
        iti_inp_silence[3, 160:161, :].repeat(10, 1),
        iti_inp_silence[3, 160:, :]
    ])

    total_iti_inp = pad_sequence([
        iti_inp_silence_cond_1,
        iti_inp_silence_cond_2,
        iti_inp_silence_cond_3,
        iti_inp_silence_cond_4
    ], batch_first=True)

    
    # Cue Input
    cue_inp_dict = {}

    for cond in range(4):

        cue_inp_dict[cond] = torch.cat([
            torch.zeros(size=(int((2.1 + 0.3 * cond) / dt), 1)),
            torch.zeros(size=(extra_steps, 1)),
        ])
        #cue_inp_dict[cond][999:999+100] = 0.01

    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3]], batch_first=True)
    
    return total_iti_inp, total_cue_inp

def get_region_borders(model_type, region, hid_dim, inp_dim):
    
    if model_type == "d1d2" and region == "alm":

        start = hid_dim*5 + int(hid_dim * 0.3)
        end = hid_dim*6

    elif model_type == "d1d2" and region == "str":

        start = 0
        end = hid_dim + int(hid_dim * 0.3)

    elif model_type == "stralm" and region == "alm":

        start = hid_dim
        end = hid_dim*2 + inp_dim

    elif model_type == "stralm" and region == "str":

        start = 0
        end = hid_dim

    elif model_type == "d1" and region == "alm":

        start = hid_dim*3
        end = hid_dim*4 + inp_dim

    elif model_type == "d1" and region == "str":

        start = 0
        end = hid_dim
    
    return start, end