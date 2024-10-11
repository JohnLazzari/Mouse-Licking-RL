import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy
import scipy.io as sio
from utils import gather_inp_data, get_acts_manipulation, get_data

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 16}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 4 # set the value globally
plt.rcParams['figure.figsize'] = [10, 8]

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1451
INP_DIM = int(HID_DIM*0.1)
DT = 1e-2
CONDITION = 0
CONSTRAINED = True
INP_TYPE = "simulated"
TRIAL_EPOCH = "full"
CHECK_PATH = f"checkpoints/d1d2_full_simulated_256n_noise.1_itinoise.1_10000iters.pth"                   # Save path
INP_PATH = "data/firing_rates/ITIProj_trialPlotAll1.mat"

def get_str2thal_weights_d1d2(
    rnn, 
    constrained=True
):

    '''
        Get the weights matrices for the thalamo-striatal loop
        
        Params:
            rnn:                Multi-regional RNN containing weights
            constrained:        Whether model uses Dale's law
    '''
    
    if constrained == True:

        # Gather all sets of weights and apply Dales law
        str2str = (rnn.str2str_sparse_mask * F.hardtanh(rnn.str2str_weight_l0_hh, 1e-10, 1)) @ rnn.str2str_D
        thal2str = F.hardtanh(rnn.thal2str_weight_l0_hh, 1e-10, 1)
        str2snr = (rnn.str2snr_mask * F.hardtanh(rnn.str2snr_weight_l0_hh, 1e-10, 1)) @ rnn.str2snr_D
        str2gpe = (rnn.str2gpe_mask * F.hardtanh(rnn.str2gpe_weight_l0_hh, 1e-10, 1)) @ rnn.str2gpe_D
        gpe2stn = F.hardtanh(rnn.gpe2stn_weight_l0_hh, 1e-10, 1) @ rnn.gpe2stn_D
        stn2snr = F.hardtanh(rnn.stn2snr_weight_l0_hh, 1e-10, 1)
        snr2thal = F.hardtanh(rnn.snr2thal_weight_l0_hh, 1e-10, 1) @ rnn.snr2thal_D
        fsi2str = F.hardtanh(rnn.fsi2str_weight, 1e-10, 1) @ rnn.fsi2str_D
        thal2fsi = F.hardtanh(rnn.thal2fsi_weight, 1e-10, 1)
        fsi2fsi = F.hardtanh(rnn.fsi2fsi_weight, 1e-10, 1) @ rnn.fsi2fsi_D

    else:

        # Gather weights without Dale's Law
        str2str = rnn.str2str_weight_l0_hh
        thal2str = rnn.thal2str_weight_l0_hh
        str2snr = rnn.str2snr_weight_l0_hh
        str2gpe = rnn.str2gpe_weight_l0_hh
        gpe2stn = rnn.gpe2stn_weight_l0_hh
        stn2snr = rnn.stn2snr_weight_l0_hh
        snr2thal = rnn.snr2thal_weight_l0_hh

    # Concatenate into single weight matrix

                    # STR       FSI      GPE         STN         SNR       Thal      ALM
    W_str = torch.cat([str2str, fsi2str, rnn.zeros, rnn.zeros, rnn.zeros, thal2str], dim=1)                                     # STR
    W_fsi = torch.cat([rnn.zeros_to_fsi, fsi2fsi, rnn.zeros_to_fsi, rnn.zeros_to_fsi, rnn.zeros_to_fsi, thal2fsi], dim=1)       # FSI
    W_gpe = torch.cat([str2gpe, rnn.zeros_from_fsi, rnn.zeros, rnn.zeros, rnn.zeros, rnn.zeros], dim=1)                         # GPE
    W_stn = torch.cat([rnn.zeros, rnn.zeros_from_fsi, rnn.zeros, gpe2stn, rnn.zeros, rnn.zeros], dim=1)                         # STN
    W_snr = torch.cat([str2snr, rnn.zeros_from_fsi, rnn.zeros, stn2snr, rnn.zeros, rnn.zeros], dim=1)                           # SNR
    W_thal = torch.cat([rnn.zeros, rnn.zeros_from_fsi, rnn.zeros, rnn.zeros, snr2thal, rnn.zeros], dim=1)                       # Thal

    # Putting all weights together
    W_rec = torch.cat([W_str, W_fsi, W_gpe, W_stn, W_snr, W_thal], dim=0)
    
    return W_rec

def analysis(
    rnn, 
    str2thal_weight, 
    iti_inp, 
    cue_inp, 
    len_seq, 
    alm_start, 
    alm_end, 
    total_num_units
):
    '''
        Performs an eigendecomposition on the str2thal weight matrix
        Determines how ALM ramping and ITI input is aligned to eigenvector containing the top eigenvalue

        Params:
            rnn:                Multi-regional RNN saved from checkpoint
            str2thal_weight:    Weight matrix for thalamostriatal loop
            iti_inp:            ITI mode input to model
            cue_inp:            Cue signal
            len_seq:            list of all sequence lengths
            alm_start:          Boundaries for hn that determines where ALM starts
            alm_end:            Boundaries for hn that determines where ALM ends
            total_num_units:    Number of units in all regions
            
    '''

    # Get the eigenvalues and top eigenvector
    eigenvalues_str2thal, eigenvectors = np.linalg.eig(str2thal_weight.T)
    left_unitary_eigenvector = eigenvectors[np.argmax(eigenvalues_str2thal)]

    # Scale the input weights by the ITI input throughout the trial
    # This will capture how the magnitude changes during the trial but the direction will stay the same
    inp_values = []
    for t in range(iti_inp.shape[1]):
        str2thal_inp = (F.hardtanh(rnn.inp_weight_str, 1e-10, 1) @ iti_inp[:, t, :].T).T
        inp_values.append(str2thal_inp.detach().cpu().numpy())
    str2thal_inp = np.stack(inp_values, axis=1)

    # Gather all regions other than the striatum using zeros to combine with above input
    # This is to get it to a proper size that can be multiplied by the eigenvector
    non_str_mask = np.zeros(shape=(str2thal_inp.shape[0], str2thal_inp.shape[1], HID_DIM * 4))
    total_inp = np.concatenate([str2thal_inp, non_str_mask], axis=-1)

    # Take inner product between iti data and the top eigenvector
    iti_act_conds = np.inner(total_inp, np.real(left_unitary_eigenvector))

    # Gather initial states for the rnn
    hn = torch.zeros(size=(1, 4, total_num_units), device="cuda")
    xn = torch.zeros(size=(1, 4, total_num_units), device="cuda")
    inhib_stim = torch.zeros(size=(4, iti_inp.shape[1], total_num_units), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        act, _ = rnn(iti_inp, cue_inp, inhib_stim, hn, xn, noise=False)
    
    # Multiply ALM activity by weights going into striatum
    # This will get the input from the ALM ramp going into the striatum
    alm2alm_weights = (F.hardtanh(rnn.alm2alm_weight_l0_hh, 1e-10, 1) @ rnn.alm2alm_D).detach().cpu().numpy()
    eigenvalues_alm2alm, _ = np.linalg.eig(alm2alm_weights.T)

    alm2str_weights = (rnn.alm2str_mask * F.hardtanh(rnn.alm2str_weight_l0_hh, 1e-10, 1)).detach().cpu().numpy()
    alm_act = np.transpose(act[:, :, alm_start:alm_end].detach().cpu().numpy(), axes=(2, 1, 0))

    # Multiply by the weights for each condition
    alm_act_conds = []
    for cond in range(alm_act.shape[-1]):
        alm_act_conds.append(alm2str_weights @ alm_act[:, :, cond])

    # reshaping ALM input for inner product
    ramp_inp = np.stack(alm_act_conds, axis=2)
    ramp_inp = np.transpose(ramp_inp, axes=(2, 1, 0))

    # Similar to ITI, gather a mask for all other regions using zeros to get the correct shape
    mask = np.zeros(shape=(ramp_inp.shape[0], ramp_inp.shape[1], left_unitary_eigenvector.shape[0] - ramp_inp.shape[2]))
    ramp_inp = np.concatenate([ramp_inp, mask], axis=2)

    # Multiply the ALM ramp input by the top eigenvector
    inner_product_alm = np.inner(ramp_inp, np.real(left_unitary_eigenvector))

    # Plot data
    
    plt.plot(iti_act_conds[0, :len_seq[0]], linewidth=10)
    plt.plot(iti_act_conds[1, :len_seq[1]], linewidth=10)
    plt.plot(iti_act_conds[2, :len_seq[2]], linewidth=10)
    plt.plot(iti_act_conds[3, :len_seq[2]], linewidth=10)
    plt.xticks([])
    plt.show()

    plt.plot(inner_product_alm[0, :len_seq[0]], linewidth=10)
    plt.plot(inner_product_alm[1, :len_seq[1]], linewidth=10)
    plt.plot(inner_product_alm[2, :len_seq[2]], linewidth=10)
    plt.plot(inner_product_alm[3, :len_seq[2]], linewidth=10)
    plt.xticks([])
    plt.show()

    plt.plot(np.real(eigenvalues_str2thal))
    plt.show()

    plt.plot(np.real(eigenvalues_alm2alm))
    plt.show()

def main():
    
    # Load checkpoint and model
    checkpoint_d1d2 = torch.load(CHECK_PATH)
    rnn_d1d2 = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    rnn_d1d2.load_state_dict(checkpoint_d1d2)

    # Determine where ALM region begins
    total_num_units_d1d2 = rnn_d1d2.total_num_units
    alm_start_d1d2 = HID_DIM*5 + rnn_d1d2.fsi_size
    alm_end_d1d2 = HID_DIM*6 + rnn_d1d2.fsi_size

    # Get ramping activity
    neural_act, peak_times = get_data(DT, TRIAL_EPOCH)
    neural_act = neural_act.cuda()

    # Get input and output data
    iti_inp, cue_inp, len_seq = gather_inp_data(DT, HID_DIM, INP_PATH, TRIAL_EPOCH, peak_times, inp_type=INP_TYPE)
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()
    
    # Gather thalamo-strial weights
    str2thal_weight_d1d2 = get_str2thal_weights_d1d2(rnn_d1d2, constrained=CONSTRAINED) 
    str2thal_weight_d1d2 = str2thal_weight_d1d2.detach().cpu().numpy().astype(np.float64)

    analysis(
        rnn_d1d2, 
        str2thal_weight_d1d2, 
        iti_inp, 
        cue_inp,
        len_seq, 
        alm_start_d1d2, 
        alm_end_d1d2, 
        total_num_units_d1d2
    )

if __name__ == "__main__":
    main()