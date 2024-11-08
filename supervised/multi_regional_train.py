import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import mRNN
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_inp_data, get_ramp, get_masks, get_data
from losses import loss
from tqdm import tqdm

HID_DIM = 100                                                                       # Hid dim of each region
OUT_DIM = 1                                                                         # Output dim (not used)
INP_DIM = int(HID_DIM * 0.1)                                                          # Input dimension
EPOCHS = 10000                                                                       # Training iterations
LR = 1e-4                                                                           # Learning rate
DT = 1e-2                                                                           # DT to control number of timesteps
WEIGHT_DECAY = 1e-3                                                                 # Weight decay parameter
MODEL_TYPE = "d1d2"                                                                 # d1d2, d1, stralm, d1d2_simple
CONSTRAINED = True                                                                  # Whether or not the model uses plausible circuit
TRIAL_EPOCH = "delay"
NMF = False
N_COMPONENTS = 5
OUT_TYPE = "ramp"
SAVE_PATH = f"checkpoints/{MODEL_TYPE}_full_100n_almnoise.01_10000iters_newloss.pth"                   # Save path

# Model with gaussian ramps looks best with 0.8, .5, .8 tonic levels, cue input to thal, and no fsi (new baseline model for simple ramping task)

def main():

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    rnn = mRNN(INP_DIM, OUT_DIM, noise_level_act=0.1, noise_level_inp=0.05, constrained=CONSTRAINED).cuda()
        
    criterion = nn.MSELoss()

    # Get ramping activity
    if OUT_TYPE == "ramp":

        neural_act, peak_times = get_ramp(dt=DT)
        neural_act = neural_act.cuda()

    elif OUT_TYPE == "data":

        neural_act, peak_times = get_data(
            DT,
            TRIAL_EPOCH,
            NMF,
            N_COMPONENTS
        )

        neural_act = neural_act.cuda()

    # Get input and output data
    iti_inp, cue_inp, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM, peaks=peak_times)
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    hn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()
    xn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()

    loss_mask_act = get_masks(OUT_DIM, len_seq)
    inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], rnn.total_num_units), device="cuda")

    best_steady_state = np.inf
    prev_steady_state = np.inf
    
    ###########################
    #    Begin Training       # 
    ###########################
    
    for epoch in range(EPOCHS):
        
        # Pass through RNN
        _, out = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=True)

        # Get masks
        out = out * loss_mask_act

        # Get loss
        loss_val = loss(
            criterion, 
            out, 
            neural_act, 
        )

        # Save model
        if epoch > 500:
            torch.save(rnn.state_dict(), SAVE_PATH)

        if epoch % 10 == 0:
            print("Training loss at epoch {}:{}".format(epoch, loss_val.item()))
            print("")

        #simple_loss = simple_dynamics_d1d2(act, rnn, HID_DIM)
        #loss += simple_loss

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss_val.backward()

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
