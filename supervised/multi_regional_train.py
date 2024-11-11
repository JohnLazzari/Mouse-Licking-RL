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
import config
from datetime import datetime

# Model with gaussian ramps looks best with 0.8, .5, .8 tonic levels, cue input to thal, and no fsi (new baseline model for simple ramping task)

def main():

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    # datetime object containing current date and time
    now = datetime.now()

    # dd_mm_YY_H_M_S for saving model
    dt_string = now.strftime("%d_%m_%Y_%H_%M_%S")

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    rnn = CBGTCL(
        args.config_file, 
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

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    hn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()
    xn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()

    loss_mask_act = get_masks(args.out_dim, len_seq)
    inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], rnn.total_num_units), device="cuda")

    best_val_loss = np.inf

    cur_loss = 0
    mean_training_loss = []
    
    ###########################
    #    Begin Training       # 
    ###########################
    
    for epoch in range(args.epochs):
        
        # Pass through RNN
        _, out = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=True)

        # Get masks
        out = out * loss_mask_act

        # Get loss
        loss_ = loss(
            criterion, 
            out, 
            neural_act, 
        )

        cur_loss += loss_.item()

        # Save model
        if epoch > 500:
            torch.save(rnn.state_dict(), args.save_path + dt_string)

        if epoch % 10 == 0:
            mean_loss = cur_loss / 10
            cur_loss = 0
            print("Mean training loss at epoch {}:{}".format(epoch, mean_loss))
            print("")

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss_.backward()

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
