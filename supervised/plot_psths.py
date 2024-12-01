import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import CBGTCL
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from utils import gather_inp_data, get_acts_control, get_acts_manipulation, load_variables_from_file, convert_value, get_ramp, get_data, get_masks, gather_train_val_test_split
import tqdm
import time
import ast
import json
import math
import os

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 12}
plt.rcParams['figure.figsize'] = [16, 8]
plt.rcParams['axes.linewidth'] = 4 # set the value globally
plt.rc('font', **font)

MODEL_NAMES = ["01_12_2024_15_45_37"]
SPECIFICATIONS_PATH = "checkpoints/model_specifications/"
CHECK_PATH = "checkpoints/"
SAVE_NAME_PATH = "results/psths/"
DATA_SPLIT = "training"
START_SILENCE = 160
END_SILENCE = 220
STIM_STRENGTH = -5
EXTRA_STEPS = 100
REGIONS_CELL_TYPES = [("alm", "exc")]

def gather_psths(
            rnn, 
            data,
            type="control"
        ):
    
    if type == "control":

        # activity without silencing
        act = get_acts_control(
            rnn, 
            data,
        )

    elif type == "manipulation": 

        # activity with silencing
        act = get_acts_manipulation(
            rnn, 
            START_SILENCE,
            END_SILENCE,
            STIM_STRENGTH,
            EXTRA_STEPS,
            REGIONS_CELL_TYPES,
            data,
        )
        
    activity_list = []
    for region in rnn.region_dict:
        if len(rnn.region_dict[region].cell_type_info) > 0:
            for cell_type in rnn.region_dict[region].cell_type_info:
                mean_act = np.mean(rnn.get_region_activity(region, act, cell_type=cell_type), axis=-1)
                activity_list.append(mean_act)
        else:
            mean_act = np.mean(rnn.get_region_activity(region, act), axis=-1)
            activity_list.append(mean_act)
    
    return activity_list

def plot_psths(psths, n_rows, n_cols, save_path):

    fig, ax = plt.subplots(n_rows, n_cols)

    psth_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if psth_idx < len(psths):
                ax[i, j].plot(psths[psth_idx].T, linewidth=5)
            psth_idx += 1

    # Get the directory part of the save path
    directory = os.path.dirname(save_path)

    # Check if the directory exists, and create it if it doesn't
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(save_path)

def main():

    # Loop through each model specified
    for model in MODEL_NAMES:

        # Open json for mRNN configuration file
        config_name = SPECIFICATIONS_PATH + model + ".json"
        
        # Load in variables from the training specs txt file
        train_specs = load_variables_from_file(SPECIFICATIONS_PATH + model + ".txt")

        # Unload variables
        inp_dim = train_specs["inp_dim"]
        out_dim = train_specs["out_dim"]
        epochs = train_specs["epochs"]
        lr = train_specs["lr"]
        dt = train_specs["dt"]
        weight_decay = train_specs["weight_decay"]
        constrained = train_specs["constrained"]
        trial_epoch = train_specs["trial_epoch"]
        nmf = train_specs["nmf"]
        n_components = train_specs["n_components"]
        out_type = train_specs["out_type"]

        # Load saved model
        checkpoint = torch.load(CHECK_PATH + model + ".pth")
        rnn = CBGTCL(config_name, inp_dim, out_dim, out_type, constrained=constrained).cuda()
        rnn.load_state_dict(checkpoint)

        # Get ramping activity
        if out_type == "ramp":

            neural_act, peak_times = get_ramp(dt=dt)
            neural_act = neural_act.cuda()

        elif out_type == "data":

            neural_act, peak_times = get_data(
                dt,
                trial_epoch,
                nmf,
                n_components
            )

            neural_act = neural_act.cuda()

        # Get input and output data
        iti_inp, cue_inp, len_seq = gather_inp_data(inp_dim=inp_dim, peaks=peak_times)
        iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()
        loss_mask = get_masks(out_dim, len_seq)

        data_dict = gather_train_val_test_split(neural_act, peak_times, iti_inp, cue_inp, loss_mask)

        activity_list_control = gather_psths(
            rnn.mrnn, 
            data_dict[DATA_SPLIT],
        )

        activity_list_manipulation = gather_psths(
            rnn.mrnn, 
            data_dict[DATA_SPLIT],
            type="manipulation"
        )
        
        plot_psths(activity_list_control, 2, 5, SAVE_NAME_PATH + model + "/" + "control.png")
        plot_psths(activity_list_manipulation, 2, 5, SAVE_NAME_PATH + model + "/" + "manipulation.png")

if __name__ == "__main__":
    main()