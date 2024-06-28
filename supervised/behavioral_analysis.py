import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_inp_data, get_acts_control, get_acts_manipulation
import tqdm

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 26}
plt.rcParams['figure.figsize'] = [10, 8]
plt.rc('font', **font)

HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDS = 3
MODEL_TYPE = "d1d2" # d1d2, d1, stralm
REGION_PERTURB = "alm" # alm, str, str_d2 (str will always be d1 for d1d2 model, specify d2 if necessary)
STIM_STRENGTH = -10
ITI_STEPS = 1000
EXTRA_STEPS_CONTROL = 0
EXTRA_STEPS_SILENCING = 2000
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}.pth"

# TODO, specify cut-off for no lick trials during control and silencing (like maybe 1s after target time so 1000 steps), make figures nicer
# automatically save instead of manually saving

def get_lick_samples(rnn, x_data, model_type, start_time, len_seq, num_samples=10):

    decision_times = {}
    # Initialize lists for conditions
    for cond in range(3):
        decision_times[cond] = []
    num_trials = 0

    if model_type == "d1d2":
        alm_start = HID_DIM*5
    elif model_type == "d1":
        alm_start = HID_DIM*2
    elif model_type == "stralm":
        alm_start = HID_DIM

    for sample in tqdm.tqdm(range(num_samples)):

        for cond in range(len(len_seq)):

            acts = get_acts_control(len_seq,
                                    rnn,
                                    HID_DIM,
                                    x_data,
                                    cond,
                                    model_type,
                                    ITI_STEPS,
                                    EXTRA_STEPS_CONTROL 
                                    ) 

            alm_acts = np.mean(acts[start_time:, alm_start:], axis=1)
            num_trials = num_trials + 1
            
            for i, act in enumerate(alm_acts):
                if act > 0.95:
                    decision_times[cond].append((i + start_time) * DT)
                    break
    
    return decision_times, num_trials

def get_lick_samples_perturbation(rnn, x_data, model_type, len_seq, region, start_silence, end_silence, start_time, num_samples=10):
    
    decision_times = {}
    for cond in range(3):
        decision_times[cond] = []
    num_trials = 0

    if model_type == "d1d2":
        alm_start = HID_DIM*5
    elif model_type == "d1":
        alm_start = HID_DIM*2
    elif model_type == "stralm":
        alm_start = HID_DIM

    for sample in tqdm.tqdm(range(num_samples)):
    
        for cond in range(len(len_seq)):

            acts = get_acts_manipulation(len_seq,
                                        rnn,
                                        HID_DIM,
                                        x_data,
                                        cond,
                                        MODEL_TYPE,
                                        ITI_STEPS,
                                        start_silence,
                                        end_silence,
                                        STIM_STRENGTH,
                                        EXTRA_STEPS_SILENCING,
                                        region
                                        )
            
            extended_len_seq = len_seq[cond] + EXTRA_STEPS_SILENCING
            alm_acts = np.mean(acts[start_time:, alm_start:], axis=1)

            num_trials = num_trials + 1

            for i, act in enumerate(alm_acts):
                if act > 0.95:
                    decision_times[cond].append((i + start_time) * DT )
                    break
    
    return decision_times, extended_len_seq, num_trials

def calculate_ecdf(decision_times, len_seq, num_trials, start_time_s):
    
    pooled_decision_times = np.concatenate([np.array(decision_times[0]), 
                                            np.array(decision_times[1]), 
                                            np.array(decision_times[2])])

    bins = np.linspace(start_time_s, len_seq * DT, len_seq - int(start_time_s / DT))
    bin_probs = []

    for bin in bins:
        prob = 0
        for time in pooled_decision_times:
            if time < bin:
                prob = prob + 1
        # Not considering no lick trials for now, since it might not matter in this case
        prob = prob / pooled_decision_times.shape[0]
        bin_probs.append(prob)

    return bins, bin_probs

def plot_ecdf(bins, bin_probs, bins_perturb, bin_probs_perturb, use_label=False):
    
    plt.plot(bins - 1.0, bin_probs, linewidth=8, color="black")
    plt.plot(bins_perturb - 1.0, bin_probs_perturb, linewidth=8, color=(0.1, 0.1, 0.75))
    plt.xlabel("Lick Time (s)")
    plt.ylabel("Proportion of Trials (CDF)")
    #plt.yticks([])
    #plt.xticks([])
    plt.show()

def plot_lick_time_difference(decision_times_control_late, decision_times_control_early, decision_times_delay_silence, decision_times_precue_silence, num_samples=1000):
    # Plots the change in lick time from precue to control or delay to control
    lick_changes_precue = []
    lick_changes_delay = []
    for cond in range(3):

        for sample in range(num_samples):

            control_sample_late = np.random.choice(decision_times_control_late[cond])
            control_sample_early = np.random.choice(decision_times_control_early[cond])

            delay_sample = np.random.choice(decision_times_delay_silence[cond])
            precue_sample = np.random.choice(decision_times_precue_silence[cond])

            delay_change = delay_sample - control_sample_late
            precue_change = precue_sample - control_sample_early

            lick_changes_precue.append(precue_change)
            lick_changes_delay.append(delay_change)
    
    plt.boxplot([lick_changes_precue, lick_changes_delay], labels=["Precue", "Delay"], )
    plt.ylabel("Change in Lick time (stim - control)")
    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.05).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.05).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.05).cuda()

    rnn.load_state_dict(checkpoint)

    x_data, len_seq = gather_inp_data(dt=0.001, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    # Control
    decision_times_control_late, num_trials_control_late = get_lick_samples(rnn, x_data, MODEL_TYPE, 1600, len_seq)
    decision_times_control_early, num_trials_control_early = get_lick_samples(rnn, x_data, MODEL_TYPE, 500, len_seq)
    # Delay Silencing
    decision_times_delay_silence, len_seq_delay_silence, num_trials_delay_silence = get_lick_samples_perturbation(rnn, 
                                                                                                                x_data, 
                                                                                                                MODEL_TYPE, 
                                                                                                                len_seq, 
                                                                                                                REGION_PERTURB, 
                                                                                                                start_silence=1600, 
                                                                                                                end_silence=2100, 
                                                                                                                start_time=1600)
    # Precue Silencing
    decision_times_precue_silence, len_seq_precue_silence, num_trials_precue_silence = get_lick_samples_perturbation(rnn, 
                                                                                                                    x_data, 
                                                                                                                    MODEL_TYPE, 
                                                                                                                    len_seq, 
                                                                                                                    REGION_PERTURB, 
                                                                                                                    start_silence=500, 
                                                                                                                    end_silence=1000, 
                                                                                                                    start_time=500)

    # Control CDF
    bins_control_late, bin_probs_control_late = calculate_ecdf(decision_times_control_late, len_seq_delay_silence, num_trials_control_late, 1.6)
    # Control CDF
    bins_control_early, bin_probs_control_early = calculate_ecdf(decision_times_control_early, len_seq_delay_silence, num_trials_control_early, 0.5)
    # Delay Silencing CDF
    bins_delay_silence, bin_probs_delay_silence = calculate_ecdf(decision_times_delay_silence, len_seq_delay_silence, num_trials_delay_silence, 1.6)
    # Precue Silencing CDF
    bins_precue_silence, bin_probs_precue_silence = calculate_ecdf(decision_times_precue_silence, len_seq_precue_silence, num_trials_precue_silence, 0.5)

    # Plot Delay Silencing CDF
    plot_ecdf(bins_control_late, bin_probs_control_late, bins_delay_silence, bin_probs_delay_silence, use_label=True)
    # Plot Delay Silencing CDF
    plot_ecdf(bins_control_early, bin_probs_control_early, bins_precue_silence, bin_probs_precue_silence)

    # Plot the lick time difference
    plot_lick_time_difference(decision_times_control_late, decision_times_control_early, decision_times_delay_silence, decision_times_precue_silence)
    
if __name__ == "__main__":
    main()