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

def gather_inp_data(dt, hid_dim):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''
    
    inp = {}

    # Condition 1: 0.3 for 1.1s
    inp[0] = torch.cat([
        0.03*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.3*torch.ones(size=(int(1.1 / dt), int(hid_dim*0.1)))
        ])

    # Condition 2: 0.6 for 1.6
    inp[1] = torch.cat([
        0.02*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.2*torch.ones(size=(int(1.6 / dt), int(hid_dim*0.1)))
        ])

    # Condition 3: 0.9 for 2.1s
    inp[2] = torch.cat([
        0.01*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
        0.1*torch.ones(size=(int(2.1 / dt), int(hid_dim*0.1)))
        ])

    # Combine all inputs
    total_inp = pad_sequence([inp[0], inp[1], inp[2]], batch_first=True)

    # Combine all sequence lengths
    len_seq = [int(2.1 / dt), int(2.6 / dt), int(3.1 / dt)]

    return total_inp, len_seq

def get_event_target(dt):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    
    event_targets = {}

    for cond in range(3):

        event_targets[cond] = torch.zeros(size=(int((1.1 + 0.5 * cond) / dt), 1))
        event_targets[cond][-1] = 1

    event_targets = pad_sequence([event_targets[0], event_targets[1], event_targets[2]], batch_first=True)

    return event_targets

def get_ramp(dt, type="None"):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    
    alm_ramps = {}
    str_ramps = {}
    thal_ramps = {}

    baseline = torch.zeros(size=(int(1.0 / dt),), dtype=torch.float32).unsqueeze(1)

    if type == "None":
        alm_mag = np.ones(shape=(3,))
        str_mag = np.ones(shape=(3,))
        thal_mag = np.ones(shape=(3,))
    elif type == "randincond":
        alm_mag = np.repeat(np.random.uniform(0.25, 1), 3)
        str_mag = np.repeat(np.random.uniform(0.25, 1), 3)
        thal_mag = np.repeat(np.random.uniform(0.25, 1), 3)
    elif type == "randacrosscond":
        thresh = np.array([1.25, 0.75, 0.25])
        alm_mag = thresh
        str_mag = thresh
        thal_mag = thresh

    for cond in range(3):

        alm_ramps[cond] = torch.cat([
            baseline,
            torch.linspace(0, alm_mag[cond], int((1.1 + 0.5 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])
        str_ramps[cond] = torch.cat([
            baseline,
            torch.linspace(0, str_mag[cond], int((1.1 + 0.5 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])
        thal_ramps[cond] = torch.cat([
            baseline,
            torch.linspace(0, thal_mag[cond], int((1.1 + 0.5 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])

    total_ramp_alm = pad_sequence([alm_ramps[0], alm_ramps[1], alm_ramps[2]], batch_first=True)
    total_ramp_str = pad_sequence([str_ramps[0], str_ramps[1], str_ramps[2]], batch_first=True)
    total_ramp_thal = pad_sequence([thal_ramps[0], thal_ramps[1], thal_ramps[2]], batch_first=True)

    return total_ramp_alm, total_ramp_str, total_ramp_thal

def get_acts_control(len_seq, rnn, hid_dim, x_data, cond, model_type, ITI_steps, extra_steps):

    '''
        Get the activities of the desired region for a single condition (silencing or activation)

        Params:
            len_seq: list of all sequence lengths for each condition
            rnn: the RNN to get activities from
            hid_dim: number of neurons in a single region
            x_data: inp data
            cond: condition to analyze (0, 1, 2)
            model_type: which pathway model to study, chooses hidden state based on this
            ITI_steps: number of ITI_steps
            start_silence: number of steps (total, starting from beginning of trial) in which to start silencing
            end_silence: number of steps in which to end silencing
            stim_strength: scale of silencing or activation
            extra_steps: number of extra steps after silencing to look at
            region: region being silenced or activated
            total_num_units: total number of units (hid_dim * num_regions)
    '''

    if model_type == "d1d2":
        hn = torch.zeros(size=(1, 1, hid_dim * 6)).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 1, hid_dim * 2)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 1, hid_dim * 4)).cuda()
    
    inhib_stim = torch.zeros(size=(1, len_seq[cond] + extra_steps, hn.shape[-1]), device="cuda")

    inp_task = x_data[cond:cond+1, :len_seq[cond], :].detach().clone()
    inp_post = x_data[cond:cond+1, ITI_steps+1:ITI_steps+2, :].repeat(1, extra_steps, 1).detach().clone()
    inp = torch.cat([inp_task, inp_post], dim=1)

    with torch.no_grad():        

        _, acts = rnn(inp, hn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_acts_manipulation(len_seq, rnn, hid_dim, x_data, cond, model_type, ITI_steps, start_silence, end_silence, stim_strength, extra_steps, region):

    '''
        Get the activities of the desired region during manipulation for a single condition (silencing or activation)

        Params:
            len_seq: list of all sequence lengths for each condition
            rnn: the RNN to get activities from
            hid_dim: number of neurons in a single region
            x_data: inp data
            cond: condition to analyze (0, 1, 2)
            model_type: which pathway model to study, chooses hidden state based on this
            ITI_steps: number of ITI_steps
            start_silence: number of steps (total, starting from beginning of trial) in which to start silencing
            end_silence: number of steps in which to end silencing
            stim_strength: scale of silencing or activation
            extra_steps: number of extra steps after silencing to look at
            region: region being silenced or activated
            total_num_units: total number of units (hid_dim * num_regions)
    '''

    if model_type == "d1d2":
        hn = torch.zeros(size=(1, 1, hid_dim * 6)).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 1, hid_dim * 2)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 1, hid_dim * 4)).cuda()
    
    inhib_stim, inp = get_inhib_stim_and_input_silence(rnn, 
                                                        region, 
                                                        x_data[cond:cond+1, :, :], 
                                                        start_silence, 
                                                        end_silence, 
                                                        len_seq[cond], 
                                                        ITI_steps, 
                                                        extra_steps, 
                                                        stim_strength, 
                                                        hn.shape[-1])
        
    with torch.no_grad():        

        _, acts = rnn(inp, hn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_masks(hid_dim, len_seq, regions):

    # mask the losses which correspond to padded values (just in case)
    loss_mask_act = [torch.ones(size=(length, hid_dim * regions), dtype=torch.int) for length in len_seq]
    loss_mask_act = pad_sequence(loss_mask_act, batch_first=True).cuda()

    return loss_mask_act

def get_ramp_mode(baseline, peak):
    
    # baseline and peak should be of shape [neurons]
    diff_vec = peak - baseline
    diff_vec = diff_vec / np.linalg.norm(diff_vec)
    return diff_vec

def project_ramp_mode(samples, ramp_mode):
    
    # samples should be [time, neurons], ramp_mode should be [neurons]
    projected = samples @ ramp_mode
    return projected

def get_inhib_stim_and_input_silence(rnn, region, x_data, start_silence, end_silence, len_seq, ITI_steps, extra_steps, stim_strength, total_num_units):

    # Select mask based on region being silenced
    if region == "alm":
        mask = rnn.alm_mask
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

    # Silencing of constant input, assuming that we are only silencing the input and never doing ALM excitation
    if region == "alm":

        inp_pre = x_data[:, :start_silence, :].detach().clone()
        inp_silence = torch.zeros(size=(1, end_silence - start_silence, x_data.shape[-1]), device="cuda")
        inp_post = x_data[:, ITI_steps+1:ITI_steps+2, :].repeat(1, (len_seq - end_silence) + extra_steps, 1).detach().clone()
        inp = torch.cat([inp_pre, inp_silence, inp_post], dim=1)
    
    elif region == "str":

        inp_pre = x_data[:, :len_seq, :].detach().clone()
        inp_post = x_data[:, ITI_steps+1:ITI_steps+2, :].repeat(1, extra_steps, 1).detach().clone()
        inp = torch.cat([inp_pre, inp_post], dim=1)
    
    return inhib_stim, inp