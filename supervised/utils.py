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
from scipy.signal import find_peaks

# creating a dictionary
font = {'size': 20}
# using rc function
plt.rc('font', **font)

def NormalizeData(
    data, 
):

    '''
        Min-Max normalization for any data

        data:       1D array of responses
    '''

    return (data - np.min(data)) / (np.max(data) - np.min(data))


def gather_inp_data(
    dt, 
    hid_dim, 
    path,
    trial_epoch,
    peaks,
    inp_type="simulated",
):

    '''
        Gather iti mode data for training
        
        Params:
            dt:             timescale in seconds (0.001 is ms)
            hid_dim:        number of hidden dimensions in a single region
            path:           path to folder containing ITI data
            trial_epoch:    Whether to use only delay epoch or full trial
            inp_type:       Whether to use simulated input or data input
            peaks:          List of peak times gathered from data for gathering delay data only
    '''

    if inp_type == "data":

        # Load in the data from the mat file and normalize the projections
        iti_projection = sio.loadmat(path)
        iti_projection = iti_projection["meanData"][:, 2500:5500:10]
        iti_projection = NormalizeData(iti_projection)

        # Hacky method right now, need to average some conditions (will get cleaner data in future)
        averaged_conds = []
        averaged_conds.append(iti_projection[0])
        averaged_conds.append(np.mean(iti_projection[1:3, :], axis=0))
        averaged_conds.append(np.mean(iti_projection[3:5, :], axis=0))
        averaged_conds.append(iti_projection[5])
        averaged_conds = np.array(averaged_conds)

        # Choose the appropriate scaling and repeat to simulate a small population of ITI neurons
        averaged_conds = torch.tensor(averaged_conds, dtype=torch.float32).unsqueeze(-1).repeat(1, 1, int(hid_dim * 0.1))
        averaged_conds[:, :100, :] *= 0.04
        averaged_conds[:, 100:, :] *= 0.4

        if trial_epoch == "full":

            # Combine all sequence lengths
            total_iti_inp = averaged_conds
            len_seq = [total_iti_inp[0].shape[0], total_iti_inp[1].shape[0], total_iti_inp[2].shape[0], total_iti_inp[3].shape[0]]

        elif trial_epoch == "delay":
            
            iti_inp_peak_cond_1 = averaged_conds[0, :peaks[0], :]
            iti_inp_peak_cond_2 = averaged_conds[1, :peaks[1], :]
            iti_inp_peak_cond_3 = averaged_conds[2, :peaks[2], :]
            iti_inp_peak_cond_4 = averaged_conds[3, :peaks[3], :]

            total_iti_inp = pad_sequence([
                iti_inp_peak_cond_1,
                iti_inp_peak_cond_2,
                iti_inp_peak_cond_3,
                iti_inp_peak_cond_4,
            ], batch_first=True)

            # Combine all sequence lengths
            len_seq = [iti_inp_peak_cond_1.shape[0], iti_inp_peak_cond_2.shape[0], iti_inp_peak_cond_3.shape[0], iti_inp_peak_cond_4.shape[0]]

    elif inp_type == "simulated":

        if trial_epoch == "delay":

            inp = {}

            # Condition 1: 1.1s
            inp[0] = torch.cat([
                0.04*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.4*torch.ones(size=(peaks[0] - 100, int(hid_dim*0.1))),
                ])

            # Condition 2: 1.4s
            inp[1] = torch.cat([
                0.03*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.3*torch.ones(size=(peaks[1] - 100, int(hid_dim*0.1))),
                ])

            # Condition 3: 1.7s
            inp[2] = torch.cat([
                0.02*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.2*torch.ones(size=(peaks[2] - 100, int(hid_dim*0.1))),
                ])

            # Condition 4: 2s
            inp[3] = torch.cat([
                0.01*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.1*torch.ones(size=(peaks[3] - 100, int(hid_dim*0.1))),
                ])

            # Combine all inputs
            total_iti_inp = pad_sequence([inp[0], inp[1], inp[2], inp[3]], batch_first=True)

            # Combine all sequence lengths
            len_seq = [peaks[0], peaks[1], peaks[2], peaks[3]]
        
        elif trial_epoch == "full":

            inp = {}

            # Condition 1: 1.1s
            inp[0] = torch.cat([
                0.04*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.4*torch.ones(size=(peaks[0] - 100, int(hid_dim*0.1))),
                torch.zeros(size=(300 - peaks[0], int(hid_dim*0.1)))
                ])

            # Condition 2: 1.4s
            inp[1] = torch.cat([
                0.03*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.3*torch.ones(size=(peaks[1] - 100, int(hid_dim*0.1))),
                torch.zeros(size=(300 - peaks[1], int(hid_dim*0.1)))
                ])

            # Condition 3: 1.7s
            inp[2] = torch.cat([
                0.02*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.2*torch.ones(size=(peaks[2] - 100, int(hid_dim*0.1))),
                torch.zeros(size=(300 - peaks[2], int(hid_dim*0.1)))
                ])

            # Condition 4: 2s
            inp[3] = torch.cat([
                0.01*torch.ones(size=(int(1.0 / dt), int(hid_dim*0.1))),
                0.1*torch.ones(size=(peaks[3] - 100, int(hid_dim*0.1))),
                torch.zeros(size=(300 - peaks[3], int(hid_dim*0.1)))
                ])

            # Combine all inputs
            total_iti_inp = pad_sequence([inp[0], inp[1], inp[2], inp[3]], batch_first=True)

            # Combine all sequence lengths
            len_seq = [total_iti_inp[0].shape[0], total_iti_inp[1].shape[0], total_iti_inp[2].shape[0], total_iti_inp[3].shape[0]]

    # Cue Input, currently not in use
    cue_inp_dict = {}
    zeros_precue = torch.zeros(size=(100, 1))

    for cond in range(4):
        cue_inp_dict[cond] = torch.cat([
            zeros_precue,
            torch.ones(size=(len_seq[cond] - 100, 1))
        ])

    # Gather all conditions for cue input
    total_cue_inp = pad_sequence([cue_inp_dict[0], cue_inp_dict[1], cue_inp_dict[2], cue_inp_dict[3]], batch_first=True)

    '''
    x = np.linspace(-1, 2, 300)
    plt.plot(x, np.mean(total_cue_inp.numpy(), axis=-1).T, c="black", linewidth=10)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    plt.show()
    '''

    return total_iti_inp, total_cue_inp, len_seq


def get_data(
    dt, 
    trial_epoch,
    pca=False, 
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

    # Normalize the data for each condition
    cond_1_alm_strain["fr_population"] = NormalizeData(cond_1_alm_strain["fr_population"])
    cond_2_alm_strain["fr_population"] = NormalizeData(cond_2_alm_strain["fr_population"])
    cond_3_alm_strain["fr_population"] = NormalizeData(cond_3_alm_strain["fr_population"])
    cond_4_alm_strain["fr_population"] = NormalizeData(cond_4_alm_strain["fr_population"])

    # Gather conditions
    neural_data_alm_strain = pad_sequence([
        torch.from_numpy(cond_1_alm_strain["fr_population"]), 
        torch.from_numpy(cond_2_alm_strain["fr_population"]), 
        torch.from_numpy(cond_3_alm_strain["fr_population"]),
        torch.from_numpy(cond_4_alm_strain["fr_population"])
    ], batch_first=True)
    
    
    ####################################
    #                                  #
    #       VLS Silencing Sessions     #
    #                                  #
    ####################################

    cond_1_str_strain["fr_population"] = NormalizeData(cond_1_str_strain["fr_population"])
    cond_2_str_strain["fr_population"] = NormalizeData(cond_2_str_strain["fr_population"])
    cond_3_str_strain["fr_population"] = NormalizeData(cond_3_str_strain["fr_population"])
    cond_4_str_strain["fr_population"] = NormalizeData(cond_4_str_strain["fr_population"])

    neural_data_str_strain = pad_sequence([
        torch.from_numpy(cond_1_str_strain["fr_population"]), 
        torch.from_numpy(cond_2_str_strain["fr_population"]), 
        torch.from_numpy(cond_3_str_strain["fr_population"]),
        torch.from_numpy(cond_4_str_strain["fr_population"])
    ], batch_first=True)


    ####################################
    #                                  #
    #       DMS Silencing Sessions     #
    #                                  #
    ####################################

    cond_1_str_dms_strain["fr_population"] = NormalizeData(cond_1_str_dms_strain["fr_population"])
    cond_2_str_dms_strain["fr_population"] = NormalizeData(cond_2_str_dms_strain["fr_population"])
    cond_3_str_dms_strain["fr_population"] = NormalizeData(cond_3_str_dms_strain["fr_population"])
    cond_4_str_dms_strain["fr_population"] = NormalizeData(cond_4_str_dms_strain["fr_population"])

    neural_data_str_dms_strain = pad_sequence([
        torch.from_numpy(cond_1_str_dms_strain["fr_population"]), 
        torch.from_numpy(cond_2_str_dms_strain["fr_population"]), 
        torch.from_numpy(cond_3_str_dms_strain["fr_population"]),
        torch.from_numpy(cond_4_str_dms_strain["fr_population"])
    ], batch_first=True)

    # Combine all sessions
    neural_data_combined = torch.cat([
        neural_data_alm_strain,
        neural_data_str_strain,
        neural_data_str_dms_strain
    ], axis=-1).type(torch.float32)

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

    if pca:
        
        # Find PCs if pca is specified
        neural_pca = PCA(n_components=n_components)
        neural_data_stacked = np.reshape(neural_data_combined, [-1, neural_data_combined.shape[-1]])
        neural_data_stacked = neural_pca.fit_transform(neural_data_stacked)
        neural_data_combined = np.reshape(neural_data_stacked, [neural_data_combined.shape[0], neural_data_combined.shape[1], n_components])

    plt.plot(np.mean(neural_data_combined.numpy(), axis=-1).T)
    plt.show()

    return neural_data_combined, peak_times


def get_acts_control(
    len_seq, 
    rnn, 
    hid_dim, 
    inp_dim, 
    iti_inp, 
    cue_inp, 
    model_type
):

    '''
        Get the activities of the desired region for all conditions during control trial

        Params:
            len_seq:            list of all sequence lengths for each condition
            rnn:                the RNN to get activities from
            hid_dim:            number of neurons in a single region
            x_data:             inp data (list: iti_inp, cue_inp)
            model_type:         which pathway model to study, chooses hidden state based on this
    '''

    # Gather initial hidden and activation states
    if model_type == "d1d2":

        hn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()
        xn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()

    elif model_type == "stralm":

        hn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()

    elif model_type == "d1":

        hn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()
    
    inhib_stim = torch.zeros(size=(4, max(len_seq), hn.shape[-1]), device="cuda")

    # Loop through network to get activities, no training performed
    with torch.no_grad():        

        acts, _ = rnn(iti_inp, cue_inp, inhib_stim, hn, xn, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    #plt.plot(np.mean(out, axis=-1).T)
    #plt.show()
    
    return acts

def get_acts_manipulation(
    len_seq, 
    rnn, 
    hid_dim, 
    inp_dim, 
    model_type, 
    start_silence, 
    end_silence, 
    stim_strength, 
    region, 
    dt,
    trial_epoch,
    peaks,
    inp_type
):

    '''
        Get the activities of the desired region during manipulation for all conditions (silencing or activation)

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

    # Gather initial hidden and activation states
    if model_type == "d1d2":

        hn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()
        xn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()

    elif model_type == "stralm":

        hn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 2 + inp_dim)).cuda()

    elif model_type == "d1":

        hn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()
        xn = torch.zeros(size=(1, 4, hid_dim * 4 + inp_dim)).cuda()

    # Inhibitory stimulus for silencing or activation
    inhib_stim = get_inhib_stim_silence(
        rnn, 
        region, 
        start_silence, 
        end_silence, 
        len_seq, 
        stim_strength
    )

    # Inputs that correspond to the silenced trial
    iti_inp_silence, cue_inp_silence = get_input_silence(
        dt, 
        region,
        hid_dim,
        start_silence,
        end_silence,
        "data/firing_rates/ITIProj_trialPlotAll1.mat",
        trial_epoch,
        peaks,
        inp_type
    )

    iti_inp_silence, cue_inp_silence = iti_inp_silence.cuda(), cue_inp_silence.cuda()
        
    # Loop through network without training
    with torch.no_grad():        

        acts, _ = rnn(iti_inp_silence, cue_inp_silence, inhib_stim, hn, xn, noise=False)
        acts = acts.squeeze().cpu().numpy()
    
    return acts

def get_masks(out_dim, len_seq):

    '''
        Get mask for training in case padded values are included in loss
        
        Params:
            out_dim:        dimension of output
            len_seq:        length of sequences
    '''

    # mask the losses which correspond to padded values (just in case)
    loss_mask_act = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
    loss_mask_act = pad_sequence(loss_mask_act, batch_first=True).cuda()

    return loss_mask_act

def get_ramp_mode(baseline, peak):

    '''
        Get the vector which is the difference between baseline activity and activity near the peak
        
        Params:
            baseline:       activity of all units before the cue, of shape [num_ramps, neurons]
            peak:           activity of all units before the peak, of shape [num_ramps, neurons]
    '''
    
    # Get the difference between the two vectors and normalize (basic form of LDA)
    diff_vec = np.mean(peak - baseline, axis=0)
    diff_vec = diff_vec / np.linalg.norm(diff_vec)

    return diff_vec

def project_ramp_mode(samples, ramp_mode):

    '''
        Take trial activity and project onto ramp mode vector
        
        Params:
            samples:        activity of all neurons throughout trial, of shape [time, neurons]
            ramp_mode:      ramp mode vector, of shape [neurons]
    '''
    
    # project activity throughout time
    projected = samples @ ramp_mode
    return projected

def get_inhib_stim_silence(rnn, region, start_silence, end_silence, len_seq, stim_strength):
    
    '''
        Get the inhibitory stimulus needed for a silencing trial
        
        Params:
            rnn:                the model being used to silence with, primarily used for masks in this function
            region:             the region being silenced, needed to determine which mask to use
            start_silence:      when silencing starts with units of 1
            end_silence:        when silencing ends with units of 1
            len_seq:            list of lengths of all of the sequences
            stim_strength:      how strong the stimulus is when applied to mask
    '''

    # Select mask based on region being silenced
    if region == "alm":
        mask = -stim_strength * (rnn.alm_ramp_mask)
    elif region == "str":
        mask = stim_strength * rnn.str_d1_mask
    elif region == "str_d2":
        mask = stim_strength * rnn.str_d2_mask
    
    # Inhibitory/excitatory stimulus to network, designed as an input current
    # Does this for a single condition, len_seq should be a single number for the chosen condition, and x_data should be [1, len_seq, :]
    inhib_stim_pre = torch.zeros(size=(4, start_silence, rnn.total_num_units), device="cuda")
    inhib_stim_silence = torch.ones(size=(4, end_silence - start_silence, rnn.total_num_units), device="cuda") * mask
    inhib_stim_post = torch.zeros(size=(4, (max(len_seq) - start_silence) + 20, rnn.total_num_units), device="cuda")
    inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)
    
    return inhib_stim

def get_input_silence(
    dt, 
    region, 
    hid_dim, 
    start_silence, 
    end_silence, 
    path,
    trial_epoch,
    peaks,
    inp_type,
):

    '''
        Gathering the input that will be needed for a silencing (or activation) trial
        
        Params:
            dt:                 time step difference
            region:             region being silenced or activated
            hid_dim:            number of units in a region
            start_silence:      when silencing starts in units of 1
            end_silence:        when silencing ends in units of 1
            path:               path to the folder containing the ITI mode data
    '''

    total_iti_inp, total_cue_inp, _ = gather_inp_data(
                                                dt, 
                                                hid_dim, 
                                                path,
                                                trial_epoch,
                                                peaks,
                                                inp_type
                                            )

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
            end = hid_dim + int(hid_dim * 0.3)

        elif region == "d1":

            start = 0
            end = int(hid_dim / 2)

        elif region == "d2":

            start = int(hid_dim / 2)
            end = hid_dim

        elif region == "fsi":

            start = hid_dim
            end = hid_dim + fsi_size

        elif region == "gpe":

            start = hid_dim + fsi_size
            end = hid_dim * 2 + fsi_size

        elif region == "stn":

            start = hid_dim * 2 + fsi_size
            end = hid_dim * 3 + fsi_size

        elif region == "snr":

            start = hid_dim * 3 + fsi_size
            end = hid_dim * 4 + fsi_size
            
        elif region == "thal":

            start = hid_dim * 4 + fsi_size
            end = hid_dim * 5 + fsi_size

        elif region == "alm_exc":

            start = hid_dim * 5 + fsi_size
            end = hid_dim * 6

        elif region == "alm_inhib":

            start = hid_dim * 6
            end = hid_dim * 6 + fsi_size

        elif region == "alm_full":

            start = hid_dim * 5 + fsi_size
            end = hid_dim * 6 + fsi_size

        elif region == "str2thal":

            start = 0
            end = hid_dim * 5 + fsi_size


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