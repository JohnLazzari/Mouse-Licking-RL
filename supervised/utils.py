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
from sklearn.decomposition import PCA, NMF
from scipy.signal import find_peaks

def NormalizeData(
    data, 
):

    '''
        Min-Max normalization for any data

        data:       1D array of responses
    '''

    return (data) / (np.percentile(data, 98) - np.percentile(data, 2) + 5)

def gaussian_density(
    x, 
    mean, 
    std
):
    return torch.exp(-(x - mean)**2/(2*std**2))

def gather_inp_data(
    inp_dim,
    peaks,
):

    '''
        Gather the input data, output target, and length of sequence for the task
        Other ramping conditions may be added in future

        dt: timescale in seconds (0.001 is ms)
    '''
    
    inp = {}

    # Condition 1: 1.1s
    inp[0] = 0.5 * torch.ones(size=(peaks[0], inp_dim))
    inp[0][:100, :] *= 0.1

    # Condition 1: 1.1s
    inp[1] = 0.4 * torch.ones(size=(peaks[1], inp_dim))
    inp[1][:100, :] *= 0.1

    # Condition 2: 1.4s
    inp[2] = 0.3 * torch.ones(size=(peaks[2], inp_dim))
    inp[2][:100, :] *= 0.1

    # Condition 3: 1.7s
    inp[3] = 0.2 * torch.ones(size=(peaks[3], inp_dim))
    inp[3][:100, :] *= 0.1

    # Condition 4: 2s
    inp[4] = 0.1 * torch.ones(size=(peaks[4], inp_dim))
    inp[4][:100, :] *= 0.1

    # Combine all inputs
    total_iti_inp = pad_sequence([inp[0], inp[1], inp[2], inp[3], inp[4]], batch_first=True)

    plt.plot(np.mean(total_iti_inp.numpy(), axis=-1).T)
    plt.show()

    # Cue Input
    cue_inp_dict = {}

    for cond in range(5):

        cue_inp_dict[cond] = torch.ones(size=(peaks[cond], 1))
        cue_inp_dict[cond][100:, :] *= 0

    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3], cue_inp_dict[4]], batch_first=True)

    # Combine all sequence lengths
    # Currently only makes sense if using delay only, which I will go with for now
    len_seq = [peaks[0], peaks[1], peaks[2], peaks[3], peaks[4]]

    return total_iti_inp, total_cue_inp, len_seq

def get_ramp(
    dt
):

    '''
        If constraining any network to a specific solution, gather the neural data or create a ramp

        dt: timescale in seconds (0.001 is ms)
    '''
    
    ramps = {}

    means = [0.8, 1.1, 1.4, 1.7, 2.0]
    std = [0.3, 0.4, 0.5, 0.6, 0.7]

    for cond in range(len(means)):

        timepoints = torch.linspace(-1, means[cond], steps=100 + int((means[cond]) / dt)).unsqueeze(-1)
        ramps[cond] = gaussian_density(timepoints, means[cond], std[cond])

    total_ramp = pad_sequence([ramps[0], ramps[1], ramps[2], ramps[3], ramps[4]], batch_first=True)
    
    peak_times = []
    for i in range(5):
        peak_times.append(100 + int(means[i] / dt))

    plt.plot(total_ramp.squeeze().numpy().T)
    plt.show()

    return total_ramp, peak_times

def get_data(
    dt, 
    trial_epoch,
    nmf=False, 
    n_components=10,
):

    '''
        Gather the target data for training, which is single unit responses in ALM
        
        Params:
            dt:                 timescale in seconds (0.001 is ms)
            pca:                whether to use PCs 
            n_components:       number of components to use for pca
            trial_epoch:        Whether to only include delay or full trial
    '''

    # Gather all data from different sessions
    
    # ALM Silencing Sessions
    cond_1_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.1s.mat")
    cond_2_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.4s.mat")
    cond_3_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.7s.mat")
    cond_4_alm_strain = sio.loadmat("data/firing_rates/alm_fr_population_2.0s.mat")

    # VLS Silencing Sessions
    cond_1_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.1s_D1silencing_control.mat")
    cond_2_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.4s_D1silencing_control.mat")
    cond_3_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.7s_D1silencing_control.mat")
    cond_4_str_strain = sio.loadmat("data/firing_rates/alm_fr_population_2.0s_D1silencing_control.mat")

    # DMS Silencing Sessions
    cond_1_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.1s_DMS_D1silencing_control.mat")
    cond_2_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.4s_DMS_D1silencing_control.mat")
    cond_3_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_1.7s_DMS_D1silencing_control.mat")
    cond_4_str_dms_strain = sio.loadmat("data/firing_rates/alm_fr_population_2.0s_DMS_D1silencing_control.mat")


    ####################################
    #                                  #
    #       ALM Silencing Sessions     #
    #                                  #
    ####################################

    # Gather conditions
    neural_data_alm_strain = pad_sequence([
        torch.from_numpy(cond_1_alm_strain["fr_population"]), 
        torch.from_numpy(cond_2_alm_strain["fr_population"]), 
        torch.from_numpy(cond_3_alm_strain["fr_population"]),
        torch.from_numpy(cond_4_alm_strain["fr_population"])
    ], batch_first=True)

    # Normalize the data for each condition
    neural_data_alm_strain = NormalizeData(neural_data_alm_strain)
    
    ####################################
    #                                  #
    #       VLS Silencing Sessions     #
    #                                  #
    ####################################

    neural_data_str_strain = pad_sequence([
        torch.from_numpy(cond_1_str_strain["fr_population"]), 
        torch.from_numpy(cond_2_str_strain["fr_population"]), 
        torch.from_numpy(cond_3_str_strain["fr_population"]),
        torch.from_numpy(cond_4_str_strain["fr_population"])
    ], batch_first=True)

    neural_data_str_strain = NormalizeData(neural_data_str_strain)


    ####################################
    #                                  #
    #       DMS Silencing Sessions     #
    #                                  #
    ####################################

    neural_data_str_dms_strain = pad_sequence([
        torch.from_numpy(cond_1_str_dms_strain["fr_population"]), 
        torch.from_numpy(cond_2_str_dms_strain["fr_population"]), 
        torch.from_numpy(cond_3_str_dms_strain["fr_population"]),
        torch.from_numpy(cond_4_str_dms_strain["fr_population"])
    ], batch_first=True)

    neural_data_str_dms_strain = NormalizeData(neural_data_str_dms_strain)

    # Combine all sessions
    neural_data_combined = torch.cat([
        neural_data_alm_strain,
        neural_data_str_strain,
        neural_data_str_dms_strain
    ], axis=-1).type(torch.float32)

    if nmf:
        
        # Find PCs if pca is specified
        neural_nmf = NMF(n_components=n_components, max_iter=10000)
        neural_data_stacked = np.reshape(neural_data_combined, [-1, neural_data_combined.shape[-1]])
        neural_data_stacked = neural_nmf.fit_transform(neural_data_stacked)
        neural_data_combined = np.reshape(neural_data_stacked, [neural_data_combined.shape[0], neural_data_combined.shape[1], n_components])
        neural_data_combined = torch.tensor(neural_data_combined, dtype=torch.float32)

    neural_data_for_peaks = np.mean(np.array(neural_data_combined), axis=-1)
    cond_1_peaks = find_peaks(neural_data_for_peaks[0])
    cond_2_peaks = find_peaks(neural_data_for_peaks[1])
    cond_3_peaks = find_peaks(neural_data_for_peaks[2])
    cond_4_peaks = find_peaks(neural_data_for_peaks[3])

    peak_times = [cond_1_peaks[0][-1], cond_2_peaks[0][-1], cond_3_peaks[0][-1], cond_4_peaks[0][-1]]

    if trial_epoch == "delay":

        neural_data_peak_cond_1 = neural_data_combined[0, :cond_1_peaks[0][-1], :]
        neural_data_peak_cond_2 = neural_data_combined[1, :cond_2_peaks[0][-1], :]
        neural_data_peak_cond_3 = neural_data_combined[2, :cond_3_peaks[0][-1], :]
        neural_data_peak_cond_4 = neural_data_combined[3, :cond_4_peaks[0][-1], :]

        # Combine all sessions
        neural_data_combined = pad_sequence([
            neural_data_peak_cond_1,
            neural_data_peak_cond_2,
            neural_data_peak_cond_3,
            neural_data_peak_cond_4
        ], batch_first=True).type(torch.float32)

    plt.plot(np.mean(neural_data_combined.numpy(), axis=-1).T)
    plt.show()

    return neural_data_combined, peak_times


def get_acts_control(rnn, data, data_split="all"):

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

    batch_size = data[data_split]["y_train"].shape[0]

    hn = torch.zeros(size=(1, batch_size, rnn.total_num_units)).cuda()
    xn = torch.zeros(size=(1, batch_size, rnn.total_num_units)).cuda()

    inhib_stim = torch.zeros(size=(4, batch_size, rnn.total_num_units), device="cuda")

    with torch.no_grad():        

        acts, _ = rnn(data[data_split]["iti_inp"], data[data_split]["cue_inp"], hn, xn, inhib_stim, noise=False)
    
    return acts

def get_acts_manipulation(rnn, start_silence, end_silence, stim_strength, extra_steps, regions_cell_types, data, data_split="all"):

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

    batch_size = data[data_split]["y_target"].shape[0]
    max_seq_len = data[data_split]["y_target"].shape[1]

    hn = torch.zeros(size=(1, batch_size, rnn.total_num_units)).cuda()
    xn = torch.zeros(size=(1, batch_size, rnn.total_num_units)).cuda()

    inhib_stim = get_inhib_stim_silence(
        rnn, 
        regions_cell_types, 
        start_silence, 
        end_silence, 
        max_seq_len, 
        extra_steps, 
        stim_strength, 
        rnn.total_num_units,
        batch_size
    )

    with torch.no_grad():        

        _, _, acts = rnn(data[data_split]["iti_inp"], data[data_split]["cue_inp"], hn, xn, inhib_stim, noise=False)
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

def get_inhib_stim_silence(rnn, regions_cell_types, start_silence, end_silence, max_seq_len, extra_steps, stim_strength, batch_size):

    """
    Get inhibitory or excitatory stimulus for optogenetic replication
    Function will gather the mask for the specified region and cell type then make a stimulus targeting these regions

    Returns:
        rnn:                        mRNN to silence
        regions_cell_types:         List of tuples specifying the region and corresponding cell type to get a mask for
        start_silence:              inteer index of when to start perturbations in the sequence
        end_silence:                integer index of when to stop perturbations in the sequence
        max_seq_len:                    max sequence length
        extra_steps:                Number of extra steps to add to the sequence if necessary
        stim_strength:              Floating point value that specifies how strong the perturbation is (- or +)
        batch_size:                 Number of conditions to be included in the sequence
    """

    # Select mask based on region being silenced
    mask = torch.zeros(size=(rnn.total_num_units))
    for region, cell_type in regions_cell_types:
        if cell_type is not None:
            cur_mask = stim_strength * (rnn.region_mask_dict[region][cell_type])
        else:
            cur_mask = stim_strength * (rnn.region_mask_dict[region]["full"])
        mask = mask + cur_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # It applies the inhibitory stimulus to all of the conditions specified in data (or max_seq_len) equally
    inhib_stim_pre = torch.zeros(size=(batch_size, start_silence, rnn.total_num_units), device="cuda")
    inhib_stim_silence = torch.ones(size=(batch_size, end_silence - start_silence, rnn.total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(batch_size, (max_seq_len - end_silence) + extra_steps, rnn.total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim
