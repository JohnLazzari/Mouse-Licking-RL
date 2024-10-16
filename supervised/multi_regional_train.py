import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_inp_data, get_ramp, get_masks, get_acts_manipulation
from losses import loss_d1d2, loss_stralm, simple_dynamics_d1d2
from tqdm import tqdm

HID_DIM = 256                                                                       # Hid dim of each region
OUT_DIM = 1                                                                         # Output dim (not used)
INP_DIM = int(HID_DIM * 0.1)                                                          # Input dimension
EPOCHS = 10000                                                                       # Training iterations
LR = 1e-4                                                                           # Learning rate
DT = 1e-2                                                                           # DT to control number of timesteps
WEIGHT_DECAY = 0                                                                 # Weight decay parameter
MODEL_TYPE = "d1d2"                                                                 # d1d2, d1, stralm, d1d2_simple
CONSTRAINED = True                                                                  # Whether or not the model uses plausible circuit
TYPE_LOSS = "alm"                                                                   # alm, threshold, none (none trains all regions to ramp, alm is just alm. alm is currently base model)
START_SILENCE = 160
END_SILENCE = 220
STIM_STRENGTH = 10
EXTRA_STEPS_SILENCE = 100
SILENCED_REGION = "alm"
SAVE_PATH = f"checkpoints/{MODEL_TYPE}_full_256n_almnoise.01_10000iters_newloss.pth"                   # Save path

'''
Default Model(s):
    HID_DIM = 100
    OUT_DIM = 1
    INP_DIM = int(HID_DIM*0.1)
    EPOCHS = 2000
    LR = 1E-4
    DT = 1E-2
    WEIGHT_DECAY = 1E-4
    MODEL_TYPE = any (d1d2, stralm, d1)
    CONSTRAINED = True
    TYPE_LOSS = alm (Only trained to ramp in alm)
    SAVE_PATH = checkpoints/d1d2_tonicsnr_fsi2str_100n_almnoise.1_itinoise.05_2000iters_newloss.pth
'''

def test(rnn, len_seq, str_start, str_end, best_steady_state):
    
    acts_manipulation = get_acts_manipulation(
        len_seq, 
        rnn, 
        HID_DIM, 
        INP_DIM,
        MODEL_TYPE, 
        START_SILENCE,
        END_SILENCE,
        STIM_STRENGTH, 
        EXTRA_STEPS_SILENCE, 
        SILENCED_REGION,
        DT 
    )

    acts_manipulation_mean = np.mean(acts_manipulation[:, :, str_start:str_end], axis=-1)
    vels = np.abs(acts_manipulation_mean[:, START_SILENCE+10:END_SILENCE] - acts_manipulation_mean[:, START_SILENCE+9:END_SILENCE-1])
    mean_vels = np.mean(vels, axis=(1, 0))

    if mean_vels < best_steady_state:
        torch.save(rnn.state_dict(), SAVE_PATH)
        best_steady_state = mean_vels
    
    return best_steady_state, mean_vels

def main():

    ####################################
    #        Training Params           #
    ####################################

    # TODO 
    # try regularization again or anything that can get it to integrate better (still mess with hyperparameters)
    # think about a better initial condition as well

    # Create RNN and specifcy objectives
    if MODEL_TYPE == "d1d2":

        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.01, noise_level_inp=0.0, constrained=CONSTRAINED).cuda()

    elif MODEL_TYPE == "d1":

        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.01, noise_level_inp=0.01, constrained=CONSTRAINED).cuda()

    elif MODEL_TYPE == "stralm":
        
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.01, noise_level_inp=0.01, constrained=CONSTRAINED).cuda()
        
    constraint_criterion = nn.MSELoss()
    thresh_criterion = nn.BCELoss()

    # Get ramping activity
    neural_act = get_ramp(dt=DT)
    neural_act = neural_act.cuda()

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM, ramp=neural_act)
    iti_inp, cue_inp = x_data
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if MODEL_TYPE == "d1d2":

        hn = torch.zeros(size=(1, 4, HID_DIM * 7 + INP_DIM + int(HID_DIM * 0.3))).cuda()
        xn = torch.zeros(size=(1, 4, HID_DIM * 7 + INP_DIM + int(HID_DIM * 0.3))).cuda()

        str_units_start = 0
        thal_units_start = HID_DIM * 5 + int(HID_DIM * 0.3)
        alm_units_start = HID_DIM * 6 + int(HID_DIM * 0.3)

        loss_mask_act = get_masks(HID_DIM, INP_DIM, len_seq, regions=7)
        inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], HID_DIM * 7 + INP_DIM + int(HID_DIM * 0.3)), device="cuda")

    elif MODEL_TYPE == "stralm":

        hn = torch.zeros(size=(1, 4, HID_DIM * 2 + INP_DIM)).cuda()
        xn = torch.zeros(size=(1, 4, HID_DIM * 2 + INP_DIM)).cuda()

        str_units_start = 0
        str_units_end = int(HID_DIM/2)
        alm_units_start = HID_DIM

        loss_mask_act = get_masks(HID_DIM, INP_DIM, len_seq, regions=2)
        inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], HID_DIM * 2 + INP_DIM), device="cuda")

    elif MODEL_TYPE == "d1":

        hn = torch.zeros(size=(1, 4, HID_DIM * 4 + INP_DIM)).cuda()
        xn = torch.zeros(size=(1, 4, HID_DIM * 4 + INP_DIM)).cuda()

        str_units_start = 0
        str_units_end = int(HID_DIM/2)
        thal_units_start = HID_DIM * 2
        alm_units_start = HID_DIM * 3

        loss_mask_act = get_masks(HID_DIM, INP_DIM, len_seq, regions=4)
        inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], HID_DIM * 4 + INP_DIM), device="cuda")
    
    best_steady_state = np.inf
    prev_steady_state = np.inf
    
    ###########################
    #    Begin Training       # 
    ###########################
    
    for epoch in range(EPOCHS):
        
        # Pass through RNN
        _, _, act = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=True)

        # Get masks
        act = act * loss_mask_act

        # Get loss
        if MODEL_TYPE == "d1d2":

            loss = loss_d1d2(
                constraint_criterion, 
                thresh_criterion,
                act, 
                neural_act, 
                HID_DIM, 
                alm_units_start, 
                str_units_start, 
                thal_units_start, 
                type=TYPE_LOSS
            )

        elif MODEL_TYPE == "stralm":

            loss = loss_stralm(
                constraint_criterion, 
                thresh_criterion,
                act, 
                neural_act, 
                alm_units_start, 
                str_units_start, 
                type=TYPE_LOSS
            )

        elif MODEL_TYPE == "d1":

            loss = loss_d1d2(
                constraint_criterion, 
                thresh_criterion,
                act, 
                neural_act, 
                HID_DIM, 
                alm_units_start, 
                str_units_start, 
                thal_units_start, 
                type=TYPE_LOSS
            )
            
        # Save model
        if epoch > 500:
            torch.save(rnn.state_dict(), SAVE_PATH)
            #best_steady_state, prev_steady_state = test(rnn, len_seq, str_units_start, str_units_end, best_steady_state)

        if epoch % 10 == 0:
            print("Training loss at epoch {}:{}".format(epoch, loss.item()))
            print("Best steady state at epoch {}:{}".format(epoch, best_steady_state))
            print("Prev steady state at epoch {}:{}".format(epoch, prev_steady_state))
            print("")

        #simple_loss = simple_dynamics_d1d2(act, rnn, HID_DIM)
        #loss += simple_loss

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss.backward()

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
