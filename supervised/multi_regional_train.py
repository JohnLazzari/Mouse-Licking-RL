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
from utils import gather_inp_data, get_ramp, get_masks
from losses import loss_d1d2, loss_stralm, simple_dynamics_d1d2, simple_dynamics_stralm, simple_dynamics_d1

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
EPOCHS = 1000
LR = 1e-4
DT = 1e-3
WEIGHT_DECAY = 1e-3
MODEL_TYPE = "d1d2" # d1d2, d1, stralm
CONSTRAINED = True
TYPE = "None" # None, randincond, randacrosscond
TYPE_LOSS = "alm" # alm or none (none trains all regions to ramp, alm is just alm. alm is currently base model)
SAVE_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}.pth"

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

def main():

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
    elif MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
        
    constraint_criterion = nn.MSELoss()

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    # Get ramping activity
    neural_act_alm, neural_act_str, neural_act_thal = get_ramp(dt=DT, type=TYPE)
    neural_act_alm, neural_act_str, neural_act_thal = neural_act_alm.cuda(), neural_act_str.cuda(), neural_act_thal.cuda()

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if MODEL_TYPE == "d1d2":

        hn = torch.zeros(size=(1, 3, HID_DIM * 6)).cuda()
        x = torch.zeros(size=(1, 3, HID_DIM * 6)).cuda()

        str_units_start = 0
        thal_units_start = HID_DIM * 4 
        alm_units_start = HID_DIM * 5

        loss_mask_act = get_masks(HID_DIM, len_seq, regions=6)
        inhib_stim = torch.zeros(size=(1, x_data.shape[1], HID_DIM*6), device="cuda")

    elif MODEL_TYPE == "stralm":

        hn = torch.zeros(size=(1, 3, HID_DIM * 2)).cuda()
        x = torch.zeros(size=(1, 3, HID_DIM * 2)).cuda()

        str_units_start = 0
        alm_units_start = HID_DIM

        loss_mask_act = get_masks(HID_DIM, len_seq, regions=2)
        inhib_stim = torch.zeros(size=(1, x_data.shape[1], HID_DIM*2), device="cuda")

    elif MODEL_TYPE == "d1":

        hn = torch.zeros(size=(1, 3, HID_DIM * 4)).cuda()
        x = torch.zeros(size=(1, 3, HID_DIM * 4)).cuda()

        str_units_start = 0
        thal_units_start = HID_DIM * 2
        alm_units_start = HID_DIM * 3

        loss_mask_act = get_masks(HID_DIM, len_seq, regions=4)
        inhib_stim = torch.zeros(size=(1, x_data.shape[1], HID_DIM*4), device="cuda")

    ###########################
    #    Begin Training       # 
    ###########################
    
    for epoch in range(EPOCHS):
        
        # Pass through RNN
        _, act, _, _ = rnn(x_data, hn, x, inhib_stim, noise=True)

        # Get masks
        act = act * loss_mask_act

        # Get loss
        if MODEL_TYPE == "d1d2":
            loss = loss_d1d2(constraint_criterion, act, neural_act_alm, neural_act_str, neural_act_thal, HID_DIM, alm_units_start, str_units_start, thal_units_start, type=TYPE_LOSS)
        elif MODEL_TYPE == "stralm":
            loss = loss_stralm(constraint_criterion, act, neural_act_alm, neural_act_str, alm_units_start, str_units_start)
        elif MODEL_TYPE == "d1":
            loss = loss_d1d2(constraint_criterion, act, neural_act_alm, neural_act_str, neural_act_thal, HID_DIM, alm_units_start, str_units_start, thal_units_start, type=TYPE_LOSS)
            
        # Save model
        if epoch > 100:
            torch.save(rnn.state_dict(), SAVE_PATH)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss.backward()

        if MODEL_TYPE == "d1d2":
            simple_dynamics_d1d2(act, rnn, HID_DIM)
        elif MODEL_TYPE == "stralm":
            simple_dynamics_stralm(act, rnn, HID_DIM)
        elif MODEL_TYPE == "d1":
            simple_dynamics_d1(act, rnn, HID_DIM)

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
