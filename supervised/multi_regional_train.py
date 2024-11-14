import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import CBGTCL
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_inp_data, get_ramp, get_masks, get_data
from losses import loss
from tqdm import tqdm
import train_config
from datetime import datetime
import json

# Model with gaussian ramps looks best with 0.8, .5, .8 tonic levels, cue input to thal, and no fsi (new baseline model for simple ramping task)
def gather_train_val_test_split(data, peak_times, iti_inp, cue_inp, loss_mask):
    
    # Conditions 1, 2, and 4 are used for training
    # Condition 3 is used for validation
    # Condition 5 is used for testing
    data_dict = {"training": {}, "validation": {}, "test": {}}
    
    data_dict["training"]["y_target"] = torch.cat([data[0], data[1], data[3]])
    data_dict["training"]["peak_times"] = [peak_times[0], peak_times[1], peak_times[3]]
    data_dict["training"]["iti_inp"] = torch.cat([iti_inp[0], iti_inp[1], iti_inp[3]])
    data_dict["training"]["cue_inp"] = torch.cat([cue_inp[0], cue_inp[1], cue_inp[3]])
    data_dict["training"]["loss_mask"] = torch.cat([loss_mask[0], loss_mask[1], loss_mask[3]])

    data_dict["validation"]["y_target"] = data[2]
    data_dict["validation"]["peak_times"] = peak_times[2]
    data_dict["validation"]["iti_inp"] = iti_inp[2]
    data_dict["validation"]["cue_inp"] = cue_inp[2]
    data_dict["validation"]["loss_mask"] = loss_mask[2]

    data_dict["test"]["y_target"] = data[4]
    data_dict["test"]["peak_times"] = peak_times[4]
    data_dict["test"]["iti_inp"] = iti_inp[4]
    data_dict["test"]["cue_inp"] = cue_inp[4]
    data_dict["test"]["loss_mask"] = loss_mask[4]

    data_dict["all"]["y_target"] = data
    data_dict["all"]["peak_times"] = peak_times
    data_dict["all"]["iti_inp"] = iti_inp
    data_dict["all"]["cue_inp"] = cue_inp
    data_dict["all"]["loss_mask"] = loss_mask

    return data_dict

def validation(rnn, val_target, val_iti_inp, val_cue_inp, mask, loss):

    hn = torch.zeros(size=(1, val_target.shape[0], rnn.total_num_units))
    xn = torch.zeros(size=(1, val_target.shape[0], rnn.total_num_units))
    inhib_stim = torch.zeros(size=(1, val_target.shape[0], rnn.total_num_units), device="cuda")

    with torch.no_grad():
        _, out = rnn(val_iti_inp, val_cue_inp, hn, xn, inhib_stim)
    
    out = out * mask
    
    val_loss = loss(out, val_target)
    val_loss = val_loss.item()
    
    return val_loss


def main():

    ### PARAMETERS ###
    parser = train_config.config_parser()
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    rnn = CBGTCL(
        args.mrnn_config_file, 
        args.inp_dim, 
        args.out_dim, 
        args.out_type, 
        noise_level_act=0.1, 
        noise_level_inp=0.05, 
        constrained=args.constrained
    ).cuda()
        
    criterion = nn.MSELoss()

    # Get ramping activity
    if args.out_type == "ramp":

        neural_act, peak_times = get_ramp(dt=args.dt)
        neural_act = neural_act.cuda()

    elif args.out_type == "data":

        neural_act, peak_times = get_data(
            args.dt,
            args.trial_epoch,
            args.nmf,
            args.n_components
        )

        neural_act = neural_act.cuda()

    # Get input and output data
    iti_inp, cue_inp, len_seq = gather_inp_data(inp_dim=args.inp_dim, peaks=peak_times)
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()
    loss_mask = get_masks(args.out_dim, len_seq)

    data_dict = gather_train_val_test_split(neural_act, peak_times, iti_inp, cue_inp, loss_mask)

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hn = torch.zeros(size=(1, data_dict["training"]["y_target"].shape[0], rnn.total_num_units)).cuda()
    xn = torch.zeros(size=(1, data_dict["training"]["y_target"].shape[0], rnn.total_num_units)).cuda()

    inhib_stim = torch.zeros(size=(1, data_dict["training"]["y_target"].shape[0], rnn.total_num_units), device="cuda")

    cur_loss = 0
    best_val_loss = np.inf

    # Load and process configuration for saving
    with open(args.mrnn_config_file, 'r') as f:
        mrnn_config = json.load(f)
    mrnn_specs = json.dumps(mrnn_config, indent=4)

    with open(args.config, "r") as file:
        training_specs = file.read()

    with open(args.model_specifications_path + dt_string, "w") as text_file:
        text_file.write("MRNN Configuration:\n")
        text_file.write("="*50 + "\n")
        text_file.write(mrnn_specs)
        text_file.write("\n\nTraining Configuration:\n")
        text_file.write("="*50 + "\n")
        text_file.write(training_specs)
    
    ###########################
    #    Begin Training       # 
    ###########################
    
    for epoch in range(args.epochs):
        
        # Pass through RNN
        _, out = rnn(data_dict["training"]["iti_inp"], data_dict["training"]["cue_inp"], hn, xn, inhib_stim, noise=True)

        # Get masks
        out = out * data_dict["training"]["loss_mask"]

        # Get loss
        loss_ = loss(
            criterion, 
            out, 
            data_dict["training"]["y_target"], 
        )

        cur_loss += loss_.item()

        if epoch % 10 == 0:

            val_loss = validation(
                rnn, 
                data_dict["validation"]["y_target"], 
                data_dict["validation"]["iti_inp"], 
                data_dict["validation"]["cue_inp"], 
                data_dict["validation"]["mask"], 
                criterion
            )

            if val_loss < best_val_loss:
                torch.save(rnn.state_dict(), args.save_path + dt_string)
                best_val_loss = val_loss

            mean_loss = cur_loss / 10
            cur_loss = 0
            print("Mean training loss at epoch {}:{}".format(epoch, mean_loss))
            print("Mean validation loss at epoch {}:{}".format(epoch, val_loss))
            print("")

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss_.backward()

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1.)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
