import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN_MultiRegional, RNN_MultiRegional_NoConstraint, RNN_MultiRegional_NoConstraintThal
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_delay_data, get_acts, get_ramp
import tqdm

CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDS = 3
MODEL_TYPE = "constraint"
REGION_PERTURB = "alm"
CONDITION = 0

def get_lick_samples(rnn, x_data, model_type, num_samples=100):

    decision_times = []
    for sample in tqdm.tqdm(range(num_samples)):

        if model_type == "constraint":
            hn = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
        elif model_type == "no_constraint":
            hn = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
        elif model_type == "no_constraint_thal":
            hn = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()

        with torch.no_grad():
            out, _, _, _, _ = rnn(x_data, hn, x, 0, noise=True)
        
        plt.plot(out[CONDITION, :, :].detach().cpu().numpy())
        plt.show()

        for i, logit in enumerate(out[CONDITION, 1200:, :]):
            num = np.random.uniform(0, 1)
            if num < logit:
                decision_times.append(i * DT + 1e-2)
                break
    
    return decision_times

def get_lick_samples_perturbation(rnn, x_data, model_type, len_seq, region, num_samples=100):
    
    # TODO, change output of behavior from sigmoid to hardtanh and regress 0 to 1 output (or keep cross entropy if that works), make silencing faster by vectorizing instead of looping

    decision_times = []

    if model_type == "constraint":
        hn = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
        x = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
    elif model_type == "no_constraint":
        hn = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
        x = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
    elif model_type == "no_constraint_thal":
        hn = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()
        x = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()

    alm_mask = rnn.alm_mask

    if model_type == "constraint":
        str_mask = rnn.str_d1_mask
    else:
        str_mask = rnn.str_mask

    ITI_steps = 1000
    extra_steps = 0
    start_silence = 600 + ITI_steps
    end_silence = 1100 + ITI_steps
    
    len_seq += (end_silence - start_silence) + extra_steps

    for sample in tqdm.tqdm(range(num_samples)):

        logits = []

        for t in range(len_seq):

            if t < ITI_steps:
                inp = x_data[CONDITION:CONDITION+1, 0:1, :]
            else:
                inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]

            with torch.no_grad():        

                if region == "alm" and t > start_silence and t < end_silence:
                    inhib_stim = -10 * alm_mask
                    inp = 0*x_data[CONDITION:CONDITION+1, 0:1, :]
                elif region == "str" and t > start_silence and t < end_silence:
                    inhib_stim = -0.25 * str_mask
                    #inhib_stim = 1 * str_mask
                else:
                    inhib_stim = 0
                        
                out, hn, _, x, _ = rnn(inp, hn, x, inhib_stim, noise=False)
                logits.append(out.squeeze().cpu().numpy())

        logits = np.array(logits)

        for i, logit in enumerate(logits[1200:]):
            num = np.random.uniform(0, 1)
            if num < logit:
                decision_times.append(i * DT + 1e-2)
                break
    
    return decision_times, len_seq

def calculate_ecdf(decision_times, len_seq):
    
    bins = np.linspace(1e-2, len_seq * DT - 1, 1000)
    bin_probs = []
    for bin in bins:
        prob = 0
        for time in decision_times:
            if time < bin:
                prob = prob + 1
        prob = prob / len(decision_times)
        bin_probs.append(prob)
    return bins, bin_probs

def plot_ecdf(bins, bin_probs, bins_perturb, bin_probs_perturb):
    
    plt.plot(bins, bin_probs, linewidth=4)
    plt.plot(bins_perturb, bin_probs_perturb, linewidth=4)
    plt.xlabel("Lick Time (s)")
    plt.ylabel("Proportion of Trials (CDF)")
    plt.show()

def main():

    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    if MODEL_TYPE == "constraint":
        rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "no_constraint":
        rnn = RNN_MultiRegional_NoConstraint(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "no_constraint_thal":
        rnn = RNN_MultiRegional_NoConstraintThal(INP_DIM, HID_DIM, OUT_DIM).cuda()

    rnn.load_state_dict(checkpoint)

    x_data, _, len_seq = gather_delay_data(dt=0.001, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    decision_times = get_lick_samples(rnn, x_data, MODEL_TYPE)
    decision_times_perturb, len_seq_perturb = get_lick_samples_perturbation(rnn, x_data, MODEL_TYPE, len_seq[CONDITION], REGION_PERTURB)

    bins, bin_probs = calculate_ecdf(decision_times, len_seq[CONDITION])
    bins_perturb, bin_probs_perturb = calculate_ecdf(decision_times_perturb, len_seq_perturb)

    plot_ecdf(bins, bin_probs, bins_perturb, bin_probs_perturb)
    
if __name__ == "__main__":
    main()