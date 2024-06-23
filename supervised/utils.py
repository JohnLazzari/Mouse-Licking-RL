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
    inp[0] = torch.cat([
        0.03*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.04))),
        0.25*torch.ones(size=(int(1.1 / dt), int(hid_dim*0.04)))
        ])

    # Condition 2: 0.6 for 1.6
    inp[1] = torch.cat([
        0.02*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.04))),
        0.2*torch.ones(size=(int(1.6 / dt), int(hid_dim*0.04)))
        ])

    # Condition 3: 0.9 for 2.1s
    inp[2] = torch.cat([
        0.01*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.04))),
        0.15*torch.ones(size=(int(2.1 / dt), int(hid_dim*0.04)))
        ])

    # Combine all inputs
    total_inp = pad_sequence([inp[0], inp[1], inp[2]], batch_first=True)

    # Lick range is 0.1s
    lick = torch.ones(size=(int(0.1 / dt), 1))

    # Condition 1: 1-1.1 lick
    no_lick = torch.zeros(size=(int(2.0 / dt), 1))
    total_lick = torch.cat([no_lick, lick], dim=0)
    lick_target[0] = total_lick

    # Condition 2: 1.5-1.6 lick
    no_lick = torch.zeros(size=(int(2.5 / dt), 1))
    total_lick = torch.cat([no_lick, lick], dim=0)
    lick_target[1] = total_lick

    # Condition 3: 2-2.1 lick
    no_lick = torch.zeros(size=(int(3.0 / dt), 1))
    total_lick = torch.cat([no_lick, lick], dim=0)
    lick_target[2] = total_lick

    # Combine all targets
    total_target = pad_sequence([lick_target[0], lick_target[1], lick_target[2]], batch_first=True)

    # Combine all sequence lengths
    len_seq = [int(2.1 / dt), int(2.6 / dt), int(3.1 / dt)]

    return total_inp, total_target, len_seq

def get_ramp(dt):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    all_ramps = {}
    baseline = torch.zeros(size=(int(1.0 / dt),), dtype=torch.float32).unsqueeze(1)
    for cond in range(3):
        all_ramps[cond] = torch.cat([
            baseline,
            torch.linspace(0, 1, int((1.1 + 0.5 * cond) / dt), dtype=torch.float32).unsqueeze(1)
        ])

    total_ramp = pad_sequence([all_ramps[0], all_ramps[1], all_ramps[2]], batch_first=True)
    return total_ramp

def get_acts(len_seq, rnn, hid_dim, x_data, cond, perturbation, model_type, region="None"):

    '''
        If silencing multi-regional model, get the activations with and without silencing
    '''

    acts = []

    alm_mask = rnn.alm_mask

    if model_type == "d1d2":
        str_mask = rnn.str_d1_mask
    else:
        str_mask = rnn.str_mask

    ITI_steps = 1000
    extra_steps_silence = 500
    extra_steps_control = 700
    start_silence = 600 + ITI_steps
    end_silence = 1100 + ITI_steps

    if model_type == "d1d2":
        hn = torch.zeros(size=(1, 1, hid_dim * 6)).cuda()
        x = torch.zeros(size=(1, 1, hid_dim * 6)).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 1, hid_dim * 2)).cuda()
        x = torch.zeros(size=(1, 1, hid_dim * 2)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 1, hid_dim * 3)).cuda()
        x = torch.zeros(size=(1, 1, hid_dim * 3)).cuda()
    
    if perturbation == True:

        if region == "alm":
            
            inhib_stim_pre = torch.zeros(size=(1, start_silence, hn.shape[-1]), device="cuda")
            inhib_stim_silence = -10 * torch.ones(size=(1, end_silence - start_silence, hn.shape[-1]), device="cuda") * alm_mask
            inhib_stim_post = torch.zeros(size=(1, (len_seq - end_silence) + (end_silence - start_silence) + extra_steps_silence, hn.shape[-1]), device="cuda")
            inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)

            inp_pre = x_data[cond:cond+1, :start_silence, :].detach().clone()
            inp_silence = torch.zeros(size=(1, end_silence - start_silence, x_data.shape[-1]), device="cuda")
            inp_post = x_data[cond:cond+1, ITI_steps+1:ITI_steps+2, :].repeat(1, (len_seq - end_silence) + (end_silence - start_silence) + extra_steps_silence, 1).detach().clone()
            inp = torch.cat([inp_pre, inp_silence, inp_post], dim=1)
        
        elif region == "str":

            inhib_stim_pre = torch.zeros(size=(1, start_silence, hn.shape[-1]), device="cuda")
            inhib_stim_silence = -0.35 * torch.ones(size=(1, end_silence - start_silence, hn.shape[-1]), device="cuda") * str_mask
            inhib_stim_post = torch.zeros(size=(1, (len_seq - end_silence) + (end_silence - start_silence) + extra_steps_silence, hn.shape[-1]), device="cuda")
            inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)

            inp_pre = x_data[cond:cond+1, :len_seq, :].detach().clone()
            inp_post = x_data[cond:cond+1, ITI_steps+1:ITI_steps+2, :].repeat(1, end_silence - start_silence + extra_steps_silence, 1).detach().clone()
            inp = torch.cat([inp_pre, inp_post], dim=1)
        
    else:

        inhib_stim = torch.zeros(size=(1, len_seq + extra_steps_control, hn.shape[-1]), device="cuda")
        inp_task = x_data[cond:cond+1, :len_seq, :].detach().clone()
        inp_post = x_data[cond:cond+1, ITI_steps+1:ITI_steps+2, :].repeat(1, extra_steps_control, 1).detach().clone()
        #inp_post = torch.zeros(size=(1, extra_steps_control, inp_task.shape[-1]), device="cuda")
        inp = torch.cat([inp_task, inp_post], dim=1)

    with torch.no_grad():        

        _, _, acts, _, _ = rnn(inp, hn, x, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_masks(out_dim, hid_dim, neural_act, len_seq, regions):

    # mask the losses which correspond to padded values (just in case)
    loss_mask = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
    loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

    loss_mask_act = [torch.ones(size=(length, hid_dim * regions), dtype=torch.int) for length in len_seq]
    loss_mask_act = pad_sequence(loss_mask_act, batch_first=True).cuda()

    loss_mask_exp = [torch.ones(size=(length, neural_act.shape[-1]), dtype=torch.int) for length in len_seq]
    loss_mask_exp = pad_sequence(loss_mask_exp, batch_first=True).cuda()

    return loss_mask, loss_mask_act, loss_mask_exp

def get_ramp_mode(baseline, peak):
    
    # baseline and peak should be of shape [neurons]
    diff_vec = peak - baseline
    diff_vec = diff_vec / np.linalg.norm(diff_vec)
    return diff_vec

def project_ramp_mode(samples, ramp_mode):
    
    # samples should be [time, neurons], ramp_mode should be [neurons]
    projected = samples @ ramp_mode
    return projected