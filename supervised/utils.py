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

def gather_inp_data(dt, hid_dim):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''
    
    inp = {}

    # Condition 1: 1.1s
    inp[0] = torch.cat([
        0.03*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.3*torch.ones(size=(int(1 / dt), int(hid_dim*0.1)))
        ])

    # Condition 2: 1.4s
    inp[1] = torch.cat([
        0.025*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.25*torch.ones(size=(int(1.5 / dt), int(hid_dim*0.1)))
        ])

    # Condition 3: 1.7s
    inp[2] = torch.cat([
        0.02*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.2*torch.ones(size=(int(2 / dt), int(hid_dim*0.1)))
        ])

    # Condition 4: 2s
    inp[3] = torch.cat([
        0.015*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.15*torch.ones(size=(int(2.5 / dt), int(hid_dim*0.1)))
        ])

    # Combine all inputs
    total_iti_inp = pad_sequence([inp[0], inp[1], inp[2], inp[3]], batch_first=True)

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

def get_ramp(dt, type="None"):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    
    alm_ramps = {}
    str_ramps = {}
    thal_ramps = {}
    baselines = {}

    baselines[0] = 0.1 * torch.ones(size=(int(1.0 / dt),), dtype=torch.float32).unsqueeze(1)
    baselines[1] = 0.1 * torch.ones(size=(int(1.0 / dt),), dtype=torch.float32).unsqueeze(1)
    baselines[2] = 0.1 * torch.ones(size=(int(1.0 / dt),), dtype=torch.float32).unsqueeze(1)
    baselines[3] = 0.1 * torch.ones(size=(int(1.0 / dt),), dtype=torch.float32).unsqueeze(1)

    if type == "None":
        alm_mag = np.ones(shape=(5,))
        str_mag = np.ones(shape=(5,))
        thal_mag = np.ones(shape=(5,))
    elif type == "randincond":
        alm_mag = np.repeat(np.random.uniform(0.25, 1), 5)
        str_mag = np.repeat(np.random.uniform(0.25, 1), 5)
        thal_mag = np.repeat(np.random.uniform(0.25, 1), 5)
    elif type == "randacrosscond":
        thresh = np.array([1.25, 0.75, 0.25, 0.25, 0.25])
        alm_mag = thresh
        str_mag = thresh
        thal_mag = thresh

    for cond in range(4):

        alm_ramps[cond] = torch.cat([
            baselines[cond],
            torch.linspace(baselines[cond][-1, 0], alm_mag[cond], int((1.1 + 0.3 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])
        str_ramps[cond] = torch.cat([
            baselines[cond],
            torch.linspace(baselines[cond][-1, 0], str_mag[cond], int((1.1 + 0.3 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])
        thal_ramps[cond] = torch.cat([
            baselines[cond],
            torch.linspace(baselines[cond][-1, 0], thal_mag[cond], int((1.1 + 0.3 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])

    total_ramp_alm = pad_sequence([alm_ramps[0], alm_ramps[1], alm_ramps[2], alm_ramps[3]], batch_first=True)
    total_ramp_str = pad_sequence([str_ramps[0], str_ramps[1], str_ramps[2], alm_ramps[3]], batch_first=True)
    total_ramp_thal = pad_sequence([thal_ramps[0], thal_ramps[1], thal_ramps[2], alm_ramps[3]], batch_first=True)

    return total_ramp_alm, total_ramp_str, total_ramp_thal

def get_acts_control(len_seq, rnn, hid_dim, inp_dim, x_data, cond, model_type, ITI_steps, extra_steps, noise=False):

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
        hn = torch.zeros(size=(1, 1, hid_dim * 6 + inp_dim)).cuda()
    elif model_type == "d1d2_simple":
        hn = torch.zeros(size=(1, 1, hid_dim * 4 + inp_dim)).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 1, hid_dim * 2 + inp_dim)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 1, hid_dim * 4 + inp_dim)).cuda()
    
    inhib_stim = torch.zeros(size=(1, len_seq[cond] + extra_steps, hn.shape[-1]), device="cuda")

    iti_inp, cue_inp = x_data 
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    inp_iti_task = iti_inp[cond:cond+1, :len_seq[cond], :].detach().clone()
    inp_iti_post = iti_inp[cond:cond+1, len_seq[cond]-1:len_seq[cond], :].repeat(1, extra_steps, 1).detach().clone()
    inp_iti = torch.cat([inp_iti_task, inp_iti_post], dim=1)

    inp_cue_task = cue_inp[cond:cond+1, :len_seq[cond], :].detach().clone()
    inp_cue_post = cue_inp[cond:cond+1, len_seq[cond]-1:len_seq[cond], :].repeat(1, extra_steps, 1).detach().clone()
    inp_cue = torch.cat([inp_cue_task, inp_cue_post], dim=1)

    with torch.no_grad():        

        _, acts = rnn(inp_iti, inp_cue, hn, inhib_stim, noise=noise)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_acts_manipulation(len_seq, rnn, hid_dim, inp_dim, x_data, cond, model_type, ITI_steps, start_silence, end_silence, stim_strength, extra_steps, region):

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
        hn = torch.zeros(size=(1, 1, hid_dim * 6 + inp_dim)).cuda()
    elif model_type == "d1d2_simple":
        hn = torch.zeros(size=(1, 1, hid_dim * 4 + inp_dim)).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 1, hid_dim * 2 + inp_dim)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 1, hid_dim * 4 + inp_dim)).cuda()

    iti_inp, cue_inp = x_data
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    inhib_stim = get_inhib_stim_silence(
        rnn, 
        region, 
        start_silence, 
        end_silence, 
        len_seq[cond], 
        extra_steps, 
        stim_strength, 
        hn.shape[-1]
    )

    iti_inp_silence, cue_inp_silence = get_input_silence(
        iti_inp[cond:cond+1], 
        cue_inp[cond:cond+1], 
        len_seq[cond], 
        extra_steps
    )
        
    with torch.no_grad():        

        _, acts = rnn(iti_inp_silence, cue_inp_silence, hn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_masks(hid_dim, inp_dim, len_seq, regions):

    # mask the losses which correspond to padded values (just in case)
    loss_mask_act = [torch.ones(size=(length, hid_dim * regions + inp_dim), dtype=torch.int) for length in len_seq]
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

def get_iti_mode(activity, lick_times):
    
    # Activity should be of shape [num_neurons, samples], lick_times [samples]
    iti_mode = []
    for neuron in activity:
        correlation = spearmanr(neuron, lick_times, axis=0)
        iti_mode.append(correlation)
    iti_mode = np.array(iti_mode)
    return iti_mode

def project_iti_mode(samples, iti_mode):
    
    # samples should be [time, neurons], iti_mode should be [neurons]
    projected = samples @ iti_mode
    return projected

def get_inhib_stim_silence(rnn, region, start_silence, end_silence, len_seq, extra_steps, stim_strength, total_num_units):

    # Select mask based on region being silenced
    if region == "alm":
        mask = rnn.full_alm_mask
    elif region == "str":
        mask = rnn.str_d1_mask
    elif region == "str_d2":
        mask = rnn.str_d2_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # Does this for a single condition, len_seq should be a single number for the chosen condition, and x_data should be [1, len_seq, :]
    inhib_stim_pre = torch.zeros(size=(1, start_silence, total_num_units), device="cuda")
    inhib_stim_silence = stim_strength * torch.ones(size=(1, end_silence - start_silence, total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(1, (len_seq - end_silence) + extra_steps, total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim

def get_input_silence(iti_inp, cue_inp, len_seq, extra_steps):

    inp_iti_pre = iti_inp[:, :len_seq, :].detach().clone()
    inp_iti_post = iti_inp[:, len_seq-1:len_seq, :].repeat(1, extra_steps, 1).detach().clone()
    inp_iti = torch.cat([inp_iti_pre, inp_iti_post], dim=1)

    inp_cue_pre = cue_inp[:, :len_seq, :].detach().clone()
    inp_cue_post = cue_inp[:, len_seq-1:len_seq, :].repeat(1, extra_steps, 1).detach().clone()
    inp_cue = torch.cat([inp_cue_pre, inp_cue_post], dim=1)
    
    return inp_iti, inp_cue

def get_region_borders(model_type, region, hid_dim, inp_dim):
    
    if model_type == "d1d2" and region == "alm":

        start = hid_dim*5
        end = hid_dim*6 + inp_dim

    elif model_type == "d1d2" and region == "str":

        start = 0
        end = int(hid_dim/2)

    elif model_type == "d1d2_simple" and region == "alm":

        start = hid_dim*3
        end = hid_dim*4 + inp_dim

    elif model_type == "d1d2_simple" and region == "str":

        start = 0
        end = int(hid_dim/2)

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