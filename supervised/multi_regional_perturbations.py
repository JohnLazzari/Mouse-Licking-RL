import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM, RNN_MultiRegional_D1
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_inp_data, get_acts_control, get_acts_manipulation, get_ramp_mode, project_ramp_mode, get_region_borders
import tqdm
import time

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 26}
plt.rcParams['figure.figsize'] = [10, 8]
plt.rcParams['axes.linewidth'] = 4 # set the value globally
plt.rc('font', **font)

HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
DT = 1e-3
CONDS = 4
MODEL_TYPE = "d1d2" # d1d2, d1, stralm
CHECK_PATH = f"checkpoints/{MODEL_TYPE}_256n_almnoise5_itinoise5_4000iters_newloss.pth"
SAVE_NAME_PATH = f"results/multi_regional_perturbations/{MODEL_TYPE}/"
CONSTRAINED = True
ITI_STEPS = 1000
START_SILENCE = 1600                    # timepoint from start of trial to silence at
END_SILENCE = 2200                      # timepoint from start of trial to end silencing
EXTRA_STEPS_SILENCE = 1000
EXTRA_STEPS_CONTROL = 0

def plot_silencing(len_seq, 
                   conds, 
                   rnn, 
                   x_data, 
                   save_name_control, 
                   save_name_silencing, 
                   silenced_region, 
                   evaluated_region, 
                   dt, 
                   stim_strength, 
                   extra_steps_control,
                   extra_steps_silence,
                   use_label=False, 
                   ):
    
    start, end = get_region_borders(MODEL_TYPE, evaluated_region, HID_DIM, INP_DIM)

    ramp_orig = {}
    ramp_silenced = {}

    # activity without silencing
    act_conds_orig = get_acts_control(
        len_seq, 
        rnn, 
        HID_DIM, 
        INP_DIM,
        x_data, 
        MODEL_TYPE
    )

    # activity with silencing
    act_conds_silenced = get_acts_manipulation(
        len_seq, 
        rnn, 
        HID_DIM, 
        INP_DIM,
        MODEL_TYPE, 
        START_SILENCE,
        END_SILENCE,
        stim_strength, 
        extra_steps_silence, 
        silenced_region,
        DT 
    )

    orig_baselines = []
    orig_peaks = []

    for cond in range(conds):

        baseline_orig_control = np.mean(act_conds_orig[cond, 500:1000, start:end], axis=0)
        peak_orig_control = np.mean(act_conds_orig[cond, 1100 + 300*cond - 200 + ITI_STEPS:1100 + 300*cond + ITI_STEPS, start:end], axis=0)

        orig_baselines.append(baseline_orig_control)
        orig_peaks.append(peak_orig_control)
    
    orig_baselines = np.array(orig_baselines)
    orig_peaks = np.array(orig_peaks)

    ramp_mode = get_ramp_mode(orig_baselines, orig_peaks)

    for cond in range(conds):

        projected_orig = project_ramp_mode(act_conds_orig[cond, :len_seq[cond] + extra_steps_control, start:end], ramp_mode)
        ramp_orig[cond] = projected_orig

        projected_silenced = project_ramp_mode(act_conds_silenced[cond, :len_seq[cond] + extra_steps_silence, start:end], ramp_mode)
        ramp_silenced[cond] = projected_silenced

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    xs_p = {}
    xs_u = {}
    for cond in range(conds):

        xs_p[cond] = np.linspace(-0.5, 1.1 + 0.3 * cond + (extra_steps_silence * dt), len_seq[cond] + extra_steps_silence - 500)
        xs_u[cond] = np.linspace(-0.5, 1.1 + 0.3 * cond + (extra_steps_control * dt), len_seq[cond] + extra_steps_control - 500)

    for cond in range(conds):
        if use_label:
            plt.plot(xs_u[cond], ramp_orig[cond][500:], label=f"Lick Time {1.1 + 0.3 * cond:.1f}s", linewidth=10)
            plt.axvline(x=1.1 + 0.3 * cond, linestyle='--')
        else:
            plt.plot(xs_u[cond], ramp_orig[cond][500:], linewidth=10)

    if use_label:
        plt.xlabel("Time (s)")
        plt.ylabel("Ramp Mode Projection")
        plt.legend(loc="lower right")
    else:
        plt.xticks([])
        
    plt.tick_params(left=False, bottom=False) 
    plt.savefig(f"{save_name_control}.png")
    plt.close()

    plt.axvline(x=0.01, linestyle='--', color='black', label="Cue")

    for cond in range(conds):
        plt.plot(xs_p[cond], ramp_silenced[cond][500:], linewidth=10)

    plt.xticks([])
    plt.tick_params(left=False, bottom=False) 
    plt.savefig(f"{save_name_silencing}.png")
    plt.close()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, constrained=CONSTRAINED).cuda()

    rnn.load_state_dict(checkpoint)

    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    
    plot_silencing(
        len_seq, 
        CONDS, 
        rnn, 
        x_data, 
        SAVE_NAME_PATH + "alm_activity_control", 
        SAVE_NAME_PATH + "alm_activity_alm_silencing",
        silenced_region="alm", 
        evaluated_region="alm", 
        dt=DT, 
        stim_strength=7, 
        extra_steps_control=EXTRA_STEPS_CONTROL,
        extra_steps_silence=EXTRA_STEPS_SILENCE,
        use_label=True
    )

    plot_silencing(
        len_seq, 
        CONDS, 
        rnn, 
        x_data, 
        SAVE_NAME_PATH + "alm_activity_control", 
        SAVE_NAME_PATH + "alm_activity_str_silencing",
        silenced_region="str", 
        evaluated_region="alm", 
        dt=DT, 
        stim_strength=-.5, 
        extra_steps_control=EXTRA_STEPS_CONTROL,
        extra_steps_silence=EXTRA_STEPS_SILENCE,
        use_label=True
    )

    plot_silencing(
        len_seq, 
        CONDS, 
        rnn, 
        x_data, 
        SAVE_NAME_PATH + "str_activity_control", 
        SAVE_NAME_PATH + "str_activity_alm_silencing",
        silenced_region="alm", 
        evaluated_region="str", 
        dt=DT, 
        stim_strength=7,
        extra_steps_control=EXTRA_STEPS_CONTROL,
        extra_steps_silence=EXTRA_STEPS_SILENCE,
        use_label=True
    )

    plot_silencing(
        len_seq, 
        CONDS, 
        rnn,
        x_data, 
        SAVE_NAME_PATH + "str_activity_control", 
        SAVE_NAME_PATH + "str_activity_str_silencing",
        silenced_region="str", 
        evaluated_region="str", 
        dt=DT, 
        stim_strength=-.5, 
        extra_steps_control=EXTRA_STEPS_CONTROL,
        extra_steps_silence=EXTRA_STEPS_SILENCE,
        use_label=True
    )
    
if __name__ == "__main__":
    main()