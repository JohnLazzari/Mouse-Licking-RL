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

def gather_inp_data(dt, hid_dim):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''
    
    inp = {}

    # Condition 1: 1.1s
    inp[0] = torch.cat([
        0.04 * torch.ones(size=(100, int(hid_dim * 0.1))),
        0.4 * torch.ones(size=(110, int(hid_dim * 0.1))),
    ])

    # Condition 2: 1.4s
    inp[1] = torch.cat([
        0.03 * torch.ones(size=(100, int(hid_dim * 0.1))),
        0.3 * torch.ones(size=(140, int(hid_dim * 0.1))),
    ])

    # Condition 3: 1.7s
    inp[2] = torch.cat([
        0.02 * torch.ones(size=(100, int(hid_dim * 0.1))),
        0.2 * torch.ones(size=(170, int(hid_dim * 0.1))),
    ])

    # Condition 4: 2s
    inp[3] = torch.cat([
        0.01 * torch.ones(size=(100, int(hid_dim * 0.1))),
        0.1 * torch.ones(size=(200, int(hid_dim * 0.1))),
    ])

    # Combine all inputs
    total_iti_inp = pad_sequence([inp[0], inp[1], inp[2], inp[3]], batch_first=True)

    #plt.plot(np.mean(total_iti_inp.numpy(), axis=-1).T)
    #plt.show()

    # Cue Input
    cue_inp_dict = {}
    cue_inp_pre_cue = torch.zeros(size=(100, 1))

    for cond in range(4):

        cue_inp_dict[cond] = torch.cat([
            cue_inp_pre_cue,
            torch.ones(size=(int((1.1 + 0.3 * cond) / dt), 1)),
        ])

    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3]], batch_first=True)

    # Combine all sequence lengths
    len_seq = [210, 240, 270, 300]

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

        '''
        ramps[cond] = torch.cat([
            torch.zeros(size=(100, 1)),
            torch.linspace(0, 1, steps=int((means[cond]) / dt)).unsqueeze(-1),
        ])
        '''

        timepoints = torch.linspace(-1, means[cond], steps=100 + int((means[cond]) / dt)).unsqueeze(-1)
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
        hn = torch.zeros(size=(1, 4, hid_dim * 7 + inp_dim + int(hid_dim * 0.3))).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 7 + inp_dim + int(hid_dim * 0.3))).cuda()
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
        hn = torch.zeros(size=(1, 4, hid_dim * 7 + inp_dim + int(hid_dim * 0.3))).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 7 + inp_dim + int(hid_dim * 0.3))).cuda()
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
        start_silence,
        end_silence, 
        region
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
        mask_inhib_units = -stim_strength * (rnn.alm_ramp_mask)
        mask_iti_units = -1 * rnn.iti_mask
        mask = mask_inhib_units + mask_iti_units
        #mask = mask_iti_units
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

def get_input_silence(dt, hid_dim, start_silence, end_silence, region):

    x_data, _ = gather_inp_data(
                        dt, 
                        hid_dim, 
                    )
    
    total_iti_inp, total_cue_inp = x_data

    if region == "alm":

        # Silence the input only during ALM silencing since it is technically ALM activity
        total_iti_inp = torch.cat([
            total_iti_inp[:, :start_silence, :],
            torch.zeros(size=(total_iti_inp.shape[0], end_silence - start_silence, total_iti_inp.shape[-1])),
            total_iti_inp[:, start_silence:start_silence+1, :].repeat(1, 20, 1),
            total_iti_inp[:, start_silence:, :]
        ], dim=1)

        total_cue_inp = torch.cat([
            total_cue_inp[:, :start_silence, :],
            torch.ones(size=(total_cue_inp.shape[0], end_silence - start_silence, total_cue_inp.shape[-1])),
            total_cue_inp[:, start_silence:start_silence+1, :].repeat(1, 20, 1),
            total_cue_inp[:, start_silence:, :]
        ], dim=1)

    #plt.plot(np.mean(total_iti_inp.numpy(), axis=-1).T)
    #plt.show()

    return total_iti_inp, total_cue_inp

def get_region_borders(
    model_type, 
    region, 
    hid_dim, 
    inp_dim
):

    '''
        Keeps track of where all of the regions are located within the hidden activity vector, returns borders 
        
        Params:
            model_type:             whether model is d1d2, d1, or stralm
            region:                 the region in question to be returned
            hid_dim:                number of neurons in a single region
            inp_dim:                number of inputs
    '''


    ###########################################
    #                                         #
    #        D1 and D2 Model Borders          #
    #                                         #
    ###########################################

    fsi_size = int(hid_dim * 0.3)

    if model_type == "d1d2":

        if region == "str":

            start = 0
            end = hid_dim * 2 + fsi_size

        elif region == "d1":

            start = 0
            end = hid_dim

        elif region == "d2":

            start = hid_dim
            end = hid_dim * 2

        elif region == "fsi":

            start = hid_dim * 2
            end = hid_dim * 2 + fsi_size

        elif region == "gpe":

            start = hid_dim * 2 + fsi_size
            end = hid_dim * 3 + fsi_size

        elif region == "stn":

            start = hid_dim * 3 + fsi_size
            end = hid_dim * 4 + fsi_size

        elif region == "snr":

            start = hid_dim * 4 + fsi_size
            end = hid_dim * 5 + fsi_size
            
        elif region == "thal":

            start = hid_dim * 5 + fsi_size
            end = hid_dim * 6 + fsi_size

        elif region == "alm_exc":

            start = hid_dim * 6 + fsi_size
            end = hid_dim * 7

        elif region == "alm_inhib":

            start = hid_dim * 7
            end = hid_dim * 7 + fsi_size

        elif region == "alm_full":

            start = hid_dim * 6 + fsi_size
            end = hid_dim * 7 + fsi_size

        elif region == "str2thal":

            start = 0
            end = hid_dim * 6 + fsi_size


    ###########################################
    #                                         #
    #          STR-ALM Model Borders          #
    #                                         #
    ###########################################

    if model_type == "stralm":
    
        if region == "alm":

            start = hid_dim
            end = hid_dim*2 + inp_dim

        elif region == "str":

            start = 0
            end = hid_dim

    return start, end 