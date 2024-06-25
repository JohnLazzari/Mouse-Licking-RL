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
from utils import gather_delay_data, get_acts, get_ramp
import tqdm

CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
HID_DIM = 256
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
DT = 1e-3
CONDS = 3
MODEL_TYPE = "d1d2" # d1d2, d1, stralm
REGION_PERTURB = "alm"
CONDITION = 2

def get_lick_samples(rnn, x_data, model_type, num_samples=10):

    decision_times = []
    for sample in tqdm.tqdm(range(num_samples)):

        if model_type == "d1d2":
            hn = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
            inhib_stim = torch.zeros(size=(1, x_data.shape[1], HID_DIM * 6), device="cuda")
        elif model_type == "stralm":
            hn = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
            inhib_stim = torch.zeros(size=(1, x_data.shape[1], HID_DIM * 2), device="cuda")
        elif model_type == "d1":
            hn = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()
            x = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()
            inhib_stim = torch.zeros(size=(1, x_data.shape[1], HID_DIM * 3), device="cuda")

        with torch.no_grad():
            out, _, acts, _, _ = rnn(x_data, hn, x, inhib_stim, noise=True)
            acts = acts.cpu().numpy()
            out = out.cpu().numpy()
        
        for i, logit in enumerate(out[CONDITION, 1200:, :]):
            num = np.random.uniform(0, 1)
            if num < logit:
                decision_times.append(i * DT + 1e-2)
                break
    
    return decision_times

def get_lick_samples_perturbation(rnn, x_data, model_type, len_seq, region, num_samples=10):
    
    decision_times = []

    if model_type == "d1d2":
        hn = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
        x = torch.zeros(size=(1, 1, HID_DIM * 6)).cuda()
    elif model_type == "stralm":
        hn = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
        x = torch.zeros(size=(1, 1, HID_DIM * 2)).cuda()
    elif model_type == "d1":
        hn = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()
        x = torch.zeros(size=(1, 1, HID_DIM * 3)).cuda()

    alm_mask = rnn.alm_mask

    if model_type == "d1d2":
        str_mask = rnn.str_d1_mask
    else:
        str_mask = rnn.str_mask

    ITI_steps = 1000
    extra_steps_silence = 500
    start_silence = 600 + ITI_steps
    end_silence = 1100 + ITI_steps

    for sample in tqdm.tqdm(range(num_samples)):

        if region == "alm":
            
            inhib_stim_pre = torch.zeros(size=(1, start_silence, hn.shape[-1]), device="cuda")
            inhib_stim_silence = -10 * torch.ones(size=(1, end_silence - start_silence, hn.shape[-1]), device="cuda") * alm_mask
            inhib_stim_post = torch.zeros(size=(1, (len_seq - end_silence) + (end_silence - start_silence) + extra_steps_silence, hn.shape[-1]), device="cuda")
            inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)

            inp_pre = x_data[CONDITION:CONDITION+1, :start_silence, :].detach().clone()
            inp_silence = torch.zeros(size=(1, end_silence - start_silence, x_data.shape[-1]), device="cuda")
            inp_post = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :].repeat(1, (len_seq - end_silence) + (end_silence - start_silence) + extra_steps_silence, 1).detach().clone()
            inp = torch.cat([inp_pre, inp_silence, inp_post], dim=1)
        
        elif region == "str":

            inhib_stim_pre = torch.zeros(size=(1, start_silence, hn.shape[-1]), device="cuda")
            inhib_stim_silence = -0.35 * torch.ones(size=(1, end_silence - start_silence, hn.shape[-1]), device="cuda") * str_mask
            inhib_stim_post = torch.zeros(size=(1, (len_seq - end_silence) + (end_silence - start_silence) + extra_steps_silence, hn.shape[-1]), device="cuda")
            inhib_stim = torch.cat([inhib_stim_pre, inhib_stim_silence, inhib_stim_post], dim=1)

            inp_pre = x_data[CONDITION:CONDITION+1, :len_seq, :].detach().clone()
            inp_post = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :].repeat(1, end_silence - start_silence + extra_steps_silence, 1).detach().clone()
            inp = torch.cat([inp_pre, inp_post], dim=1)
        
        extended_len_seq = inp.shape[1]

        with torch.no_grad():        

            logits, _, acts, _, _ = rnn(inp, hn, x, inhib_stim, noise=True)
            acts = acts.squeeze().cpu().numpy()
            logits = logits.squeeze().cpu().numpy()
            plt.plot(logits)
            plt.show()

        for i, logit in enumerate(logits[1200:]):
            num = np.random.uniform(0, 1)
            if num < logit:
                decision_times.append(i * DT + 1e-2)
                break
    
    return decision_times, extended_len_seq

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
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM).cuda()

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