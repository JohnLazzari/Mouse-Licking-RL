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
from sklearn.decomposition import PCA

def NormalizeData(data, min, max):
    '''
        Min-Max normalization for any data

        data: full array of data
        min: minimum of data along each row
        max: maximum of data along each row
    '''
    return (data - min) / (max - min)

def gather_inp_data(dt, hid_dim, path):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''

    iti_projection = sio.loadmat(path)
    iti_projection = iti_projection["meanData"][:, 2500:5500:10]
    iti_projection = NormalizeData(iti_projection, np.min(iti_projection), np.max(iti_projection))

    # Hacky method right now, need to average some conditions
    averaged_conds = []
    averaged_conds.append(iti_projection[0])
    averaged_conds.append(np.mean(iti_projection[1:3, :], axis=0))
    averaged_conds.append(np.mean(iti_projection[3:5, :], axis=0))
    averaged_conds.append(iti_projection[5])
    averaged_conds = np.array(averaged_conds)

    averaged_conds = 0.5 * torch.tensor(averaged_conds, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, int(hid_dim * 0.1))

    # Cue Input
    cue_inp_dict = {}

    for cond in range(4):

        cue_inp_dict[cond] = torch.zeros(size=(int(3 / dt), 1))
        #cue_inp_dict[cond][999:999+100] = 0.01

    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3]], batch_first=True)

    # Combine all sequence lengths
    len_seq = [int(3 / dt), int(3 / dt), int(3 / dt), int(3 / dt)]

    total_inp = [averaged_conds, total_cue_inp]

    return total_inp, len_seq

def get_data(dt, pca=False, n_components=10):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    
    cond_1_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.1s.mat")
    cond_2_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.4s.mat")
    cond_3_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.7s.mat")
    cond_4_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_2.0s.mat")

    cond_1_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.1s_D1silencing_control.mat")
    cond_2_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.4s_D1silencing_control.mat")
    cond_3_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.7s_D1silencing_control.mat")
    cond_4_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_2.0s_D1silencing_control.mat")

    cond_1_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.1s_DMS_D1silencing_control.mat")
    cond_2_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.4s_DMS_D1silencing_control.mat")
    cond_3_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.7s_DMS_D1silencing_control.mat")
    cond_4_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_2.0s_DMS_D1silencing_control.mat")

    neural_data_alm_strain = np.array([
        cond_1_alm_strain["fr_population"], 
        cond_2_alm_strain["fr_population"], 
        cond_3_alm_strain["fr_population"],
        cond_4_alm_strain["fr_population"]
    ])

    neural_data_alm_strain[0] = NormalizeData(neural_data_alm_strain[0], np.min(neural_data_alm_strain[0]), np.max(neural_data_alm_strain[0]))
    neural_data_alm_strain[1] = NormalizeData(neural_data_alm_strain[1], np.min(neural_data_alm_strain[1]), np.max(neural_data_alm_strain[1]))
    neural_data_alm_strain[2] = NormalizeData(neural_data_alm_strain[2], np.min(neural_data_alm_strain[2]), np.max(neural_data_alm_strain[2]))
    neural_data_alm_strain[3] = NormalizeData(neural_data_alm_strain[2], np.min(neural_data_alm_strain[2]), np.max(neural_data_alm_strain[2]))
    
    neural_data_str_strain = np.array([
        cond_1_str_strain["fr_population"], 
        cond_2_str_strain["fr_population"], 
        cond_3_str_strain["fr_population"],
        cond_4_str_strain["fr_population"]
    ])

    neural_data_str_dms_strain = np.array([
        cond_1_str_dms_strain["fr_population"], 
        cond_2_str_dms_strain["fr_population"], 
        cond_3_str_dms_strain["fr_population"],
        cond_4_str_dms_strain["fr_population"]
    ])

    neural_data_str_strain[0] = NormalizeData(neural_data_str_strain[0], np.min(neural_data_str_strain[0]), np.max(neural_data_str_strain[0]))
    neural_data_str_strain[1] = NormalizeData(neural_data_str_strain[1], np.min(neural_data_str_strain[1]), np.max(neural_data_str_strain[1]))
    neural_data_str_strain[2] = NormalizeData(neural_data_str_strain[2], np.min(neural_data_str_strain[2]), np.max(neural_data_str_strain[2]))
    neural_data_str_strain[3] = NormalizeData(neural_data_str_strain[3], np.min(neural_data_str_strain[3]), np.max(neural_data_str_strain[3]))

    neural_data_str_dms_strain[0] = NormalizeData(neural_data_str_dms_strain[0], np.min(neural_data_str_dms_strain[0]), np.max(neural_data_str_dms_strain[0]))
    neural_data_str_dms_strain[1] = NormalizeData(neural_data_str_dms_strain[1], np.min(neural_data_str_dms_strain[1]), np.max(neural_data_str_dms_strain[1]))
    neural_data_str_dms_strain[2] = NormalizeData(neural_data_str_dms_strain[2], np.min(neural_data_str_dms_strain[2]), np.max(neural_data_str_dms_strain[2]))
    neural_data_str_dms_strain[3] = NormalizeData(neural_data_str_dms_strain[3], np.min(neural_data_str_dms_strain[3]), np.max(neural_data_str_dms_strain[3]))

    neural_data_combined = np.concatenate([
        neural_data_alm_strain,
        neural_data_str_strain,
        neural_data_str_dms_strain
    ], axis=-1)

    if pca:
        
        neural_pca = PCA(n_components=n_components)
        neural_data_stacked = np.reshape(neural_data_combined, [-1, neural_data_combined.shape[-1]])
        neural_data_stacked = neural_pca.fit_transform(neural_data_stacked)
        neural_data_combined = np.reshape(neural_data_stacked, [neural_data_combined.shape[0], neural_data_combined.shape[1], n_components])
        
    neural_data = torch.Tensor(neural_data_combined) 
    
    return neural_data    

def get_acts_control(len_seq, rnn, hid_dim, inp_dim, x_data, model_type):

    '''
        Get the activities of the desired region for a single condition (silencing or activation)

        Params:
            len_seq:            list of all sequence lengths for each condition
            rnn:                the RNN to get activities from
            hid_dim:            number of neurons in a single region
            x_data:             inp data (list: iti_inp, cue_inp)
            model_type:         which pathway model to study, chooses hidden state based on this
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

        acts, _ = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_acts_manipulation(len_seq, rnn, hid_dim, inp_dim, model_type, start_silence, end_silence, stim_strength, region, dt):

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
        stim_strength
    )

    iti_inp_silence, cue_inp_silence = get_input_silence(
        dt, 
        hid_dim,
        start_silence,
        end_silence,
        "data/firing_rates/ITIProj_trialPlotAll1.mat"
    )

    iti_inp_silence, cue_inp_silence = iti_inp_silence.cuda(), cue_inp_silence.cuda()
        
    with torch.no_grad():        

        acts, _ = rnn(iti_inp_silence, cue_inp_silence, hn, xn, inhib_stim, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_masks(out_dim, len_seq):

    # mask the losses which correspond to padded values (just in case)
    loss_mask_act = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
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

def get_inhib_stim_silence(rnn, region, start_silence, end_silence, len_seq, stim_strength):

    # Select mask based on region being silenced
    if region == "alm":
        mask_inhib_units = -stim_strength * rnn.alm_ramp_mask
        mask_iti_units = 0 * rnn.iti_mask
        mask = mask_inhib_units + mask_iti_units
    elif region == "str":
        mask = stim_strength * rnn.str_d1_mask
    elif region == "str_d2":
        mask = stim_strength * rnn.str_d2_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # Does this for a single condition, len_seq should be a single number for the chosen condition, and x_data should be [1, len_seq, :]
    inhib_stim_pre = torch.zeros(size=(4, start_silence, rnn.total_num_units), device="cuda")
    inhib_stim_silence = torch.ones(size=(4, end_silence - start_silence, rnn.total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(4, (max(len_seq) - start_silence), rnn.total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim

def get_input_silence(dt, hid_dim, start_silence, end_silence, path):

    iti_projection = sio.loadmat(path)
    iti_projection = iti_projection["meanData"][:, 2500:5500:10]
    iti_projection = NormalizeData(iti_projection, np.min(iti_projection), np.max(iti_projection))

    # Hacky method right now, need to average some conditions
    averaged_conds = []
    averaged_conds.append(iti_projection[0])
    averaged_conds.append(np.mean(iti_projection[1:3, :], axis=0))
    averaged_conds.append(np.mean(iti_projection[3:5, :], axis=0))
    averaged_conds.append(iti_projection[5])
    averaged_conds = np.array(averaged_conds)

    inp_silence = np.concatenate([
        averaged_conds[:, :start_silence],
        np.zeros(shape=(averaged_conds.shape[0], end_silence - start_silence)),
        averaged_conds[:, start_silence:]
    ], axis=-1)

    inp_silence = 0.5 * torch.tensor(inp_silence, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, int(hid_dim * 0.1))

    # Cue Input
    cue_inp_dict = {}

    for cond in range(4):

        cue_inp_dict[cond] = torch.cat([
            torch.zeros(size=(int((2.1 + 0.3 * cond) / dt), 1)),
            torch.zeros(size=(end_silence - start_silence, 1)),
        ])
        #cue_inp_dict[cond][999:999+100] = 0.01

    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3]], batch_first=True)
    
    return inp_silence, total_cue_inp

def get_region_borders(model_type, region, hid_dim, inp_dim):
    
    if model_type == "d1d2" and region == "alm":

        # Only excitatory units
        start = hid_dim*5 + int(hid_dim * 0.3)
        end = hid_dim * 6

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