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
INP_DIM = int(HID_DIM*0.1)                                                          # Input dimension
EPOCHS = 4000                                                                       # Training iterations
LR = 1e-4                                                                           # Learning rate
DT = 1e-3                                                                           # DT to control number of timesteps
WEIGHT_DECAY = 1e-3                                                                 # Weight decay parameter
MODEL_TYPE = "d1d2"                                                                 # d1d2, d1, stralm, d1d2_simple
CONSTRAINED = True                                                                  # Whether or not the model uses plausible circuit
TYPE = "None"                                                                       # None, randincond, randacrosscond (for thresholds)
TYPE_LOSS = "alm"                                                                   # alm, threshold, none (none trains all regions to ramp, alm is just alm. alm is currently base model)
START_SILENCE = 1600
END_SILENCE = 2200
STIM_STRENGTH = 10
EXTRA_STEPS_SILENCE = 1000
SILENCED_REGION = "alm"
SAVE_PATH = f"checkpoints/{MODEL_TYPE}_256n_almnoise5_itinoise5_4000iters_newloss.pth"                   # Save path

'''
Default Model(s):
    HID_DIM = 256
    OUT_DIM = 1
    INP_DIM = int(HID_DIM*0.1)
    EPOCHS = 1000
    LR = 1E-4
    DT = 1E-3
    WEIGHT_DECAY = 1E-3
    MODEL_TYPE = any (d1d2, stralm, d1)
    CONSTRAINED = True
    TYPE = None (ramping conditions, None means ramp to one across conditions)
    TYPE_LOSS = alm (Only trained to ramp in alm)
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

    vels = acts_manipulation[:, START_SILENCE+100:END_SILENCE, str_start:str_end] - acts_manipulation[:, START_SILENCE+99:END_SILENCE-1, str_start:str_end]
    mean_vels = np.abs(np.mean(vels, axis=(1, 0, 2)))

    if mean_vels < best_steady_state:
        torch.save(rnn.state_dict(), SAVE_PATH)
        best_steady_state = mean_vels
    
    return best_steady_state, mean_vels

def main():

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=5.0, noise_level_inp=5.0, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.01, noise_level_inp=0.01, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.01, noise_level_inp=0.01, constrained=CONSTRAINED).cuda()
        
    constraint_criterion = nn.MSELoss()
    thresh_criterion = nn.BCELoss()

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    iti_inp, cue_inp = x_data
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Get ramping activity
    neural_act_alm, neural_act_str, neural_act_thal = get_ramp(dt=DT, type=TYPE)
    neural_act_alm, neural_act_str, neural_act_thal = neural_act_alm.cuda(), neural_act_str.cuda(), neural_act_thal.cuda()

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if MODEL_TYPE == "d1d2":

        hn = torch.zeros(size=(1, 4, HID_DIM * 6 + INP_DIM)).cuda()
        xn = torch.zeros(size=(1, 4, HID_DIM * 6 + INP_DIM)).cuda()

        str_units_start = 0
        str_units_end = int(HID_DIM/2)
        thal_units_start = HID_DIM * 4 
        alm_units_start = HID_DIM * 5

        loss_mask_act = get_masks(HID_DIM, INP_DIM, len_seq, regions=6)
        inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], HID_DIM * 6 + INP_DIM), device="cuda")

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
    
    for epoch in tqdm(range(EPOCHS)):
        
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
                neural_act_alm, 
                neural_act_str, 
                neural_act_thal, 
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
                neural_act_alm, 
                neural_act_str, 
                alm_units_start, 
                str_units_start, 
                type=TYPE_LOSS
            )

        elif MODEL_TYPE == "d1":

            loss = loss_d1d2(
                constraint_criterion, 
                thresh_criterion,
                act, 
                neural_act_alm, 
                neural_act_str, 
                neural_act_thal, 
                HID_DIM, 
                alm_units_start, 
                str_units_start, 
                thal_units_start, 
                type=TYPE_LOSS
            )
            
        # Save model
        if epoch > 3000:
            best_steady_state, prev_steady_state = test(rnn, len_seq, str_units_start, str_units_end, best_steady_state)

        if epoch % 10 == 0:
            print("Training loss at epoch {}:{}".format(epoch, loss.item()))
            print("Best steady state at epoch {}:{}".format(epoch, best_steady_state))
            print("Prev steady state at epoch {}:{}".format(epoch, prev_steady_state))

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss.backward()

        #if MODEL_TYPE == "d1d2":
        #    simple_dynamics_d1d2(act, rnn, HID_DIM) 

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
