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
from utils import gather_inp_data, get_ramp, get_masks, get_data, gather_train_val_test_split
from losses import loss
from tqdm import tqdm
import train_config
from datetime import datetime
import json

# Model with gaussian ramps looks best with 0.8, .5, .8 tonic levels, cue input to thal, and no fsi (new baseline model for simple ramping task)

def validation(rnn, val_target, val_iti_inp, val_cue_inp, mask, criterion):

    hn = torch.zeros(size=(1, val_target.shape[0], rnn.total_num_units), device="cuda")
    xn = torch.zeros(size=(1, val_target.shape[0], rnn.total_num_units), device="cuda")
    inhib_stim = torch.zeros(size=(val_target.shape[0], val_target.shape[1], rnn.total_num_units), device="cuda")

    with torch.no_grad():
        _, out = rnn(val_iti_inp, val_cue_inp, hn, xn, inhib_stim, noise=False)
    
    out = out * mask
    
    val_loss = criterion(out[:, 50:, :], val_target[:, 50:, :])
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
        noise_level_inp=0.1, 
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

    hn = torch.zeros(size=(1, data_dict["training"]["y_target"].shape[0], rnn.total_num_units), device="cuda")
    xn = torch.zeros(size=(1, data_dict["training"]["y_target"].shape[0], rnn.total_num_units), device="cuda")

    inhib_stim = torch.zeros(size=(data_dict["training"]["y_target"].shape[0], data_dict["training"]["y_target"].shape[1], rnn.total_num_units), device="cuda")

    cur_loss = 0
    best_val_loss = np.inf

    # Load and process mRNN configuration for specifications
    with open(args.mrnn_config_file, 'r') as f:
        mrnn_config = json.load(f)
    with open(args.model_specifications_path + dt_string + ".json", 'w') as destination_file:
        json.dump(mrnn_config, destination_file, indent=4)  # `indent=4` for pretty-printing

    # Load and process training configuration for specifications
    with open(args.config, "r") as file:
        training_specs = file.read()
    with open(args.model_specifications_path + dt_string + ".txt", "w") as text_file:
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

        torch.save(rnn.state_dict(), args.save_path + dt_string + ".pth")
        if epoch % 10 == 0 and epoch > 0:

            val_loss = validation(
                rnn, 
                data_dict["validation"]["y_target"], 
                data_dict["validation"]["iti_inp"], 
                data_dict["validation"]["cue_inp"], 
                data_dict["validation"]["loss_mask"], 
                criterion
            )

            if val_loss < best_val_loss:
                #torch.save(rnn.state_dict(), args.save_path + dt_string + ".pth")
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
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
