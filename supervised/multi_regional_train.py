import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import RNN_MultiRegional, RNN_MultiRegional_NoConstraint, RNN_MultiRegional_NoConstraintThal
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_delay_data, get_ramp, get_masks
from losses import loss_constraint, loss_no_constraint, simple_dynamics_constraint, simple_dynamics_no_constraint

SAVE_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
EPOCHS = 2000
LR = 1e-4
DT = 1e-3
WEIGHT_DECAY = 1e-4
MODEL_TYPE = "constraint" # constraint, no_constraint, no_constraint_thal

def main():

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    if MODEL_TYPE == "constraint":
        rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "no_constraint":
        rnn = RNN_MultiRegional_NoConstraint(INP_DIM, HID_DIM, OUT_DIM).cuda()
    elif MODEL_TYPE == "no_constraint_thal":
        rnn = RNN_MultiRegional_NoConstraintThal(INP_DIM, HID_DIM, OUT_DIM).cuda()
        
    criterion = nn.BCELoss()
    constraint_criterion = nn.MSELoss()

    # Get input and output data
    x_data, y_data, len_seq = gather_delay_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    y_data = y_data.cuda()
    
    # Get ramping activity
    neural_act = get_ramp(dt=DT)
    neural_act = neural_act.cuda()

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ####################################
    #          Train RNN               #
    ####################################

    if MODEL_TYPE == "constraint":

        hn = torch.zeros(size=(1, 3, HID_DIM * 6)).cuda()
        x = torch.zeros(size=(1, 3, HID_DIM * 6)).cuda()

        str_units_start = 0
        thal_units_start = HID_DIM * 4 
        alm_units_start = HID_DIM * 5

        loss_mask, loss_mask_act, loss_mask_exp = get_masks(OUT_DIM, HID_DIM, neural_act, len_seq, regions=6)

    elif MODEL_TYPE == "no_constraint":

        hn = torch.zeros(size=(1, 3, HID_DIM * 2)).cuda()
        x = torch.zeros(size=(1, 3, HID_DIM * 2)).cuda()

        str_units_start = 0
        alm_units_start = HID_DIM

        loss_mask, loss_mask_act, loss_mask_exp = get_masks(OUT_DIM, HID_DIM, neural_act, len_seq, regions=2)

    elif MODEL_TYPE == "no_constraint_thal":

        hn = torch.zeros(size=(1, 3, HID_DIM * 3)).cuda()
        x = torch.zeros(size=(1, 3, HID_DIM * 3)).cuda()

        str_units_start = 0
        thal_units_start = HID_DIM 
        alm_units_start = HID_DIM * 2

        loss_mask, loss_mask_act, loss_mask_exp = get_masks(OUT_DIM, HID_DIM, neural_act, len_seq, regions=3)

    best_loss = np.inf

    for epoch in range(EPOCHS):
        
        # Pass through RNN
        out, _, act, _, _ = rnn(x_data, hn, x, 0, noise=True)

        # Get masks
        out = out * loss_mask
        act = act * loss_mask_act
        neural_act = neural_act * loss_mask_exp

        # Get loss
        if MODEL_TYPE == "constraint":
            loss = loss_constraint(criterion, constraint_criterion, act, out, neural_act, y_data, HID_DIM, alm_units_start, str_units_start, thal_units_start)
        elif MODEL_TYPE == "no_constraint":
            loss = loss_no_constraint(constraint_criterion, act, neural_act, alm_units_start, str_units_start)
        elif MODEL_TYPE == "no_constraint_thal":
            loss = loss_constraint(constraint_criterion, act, out, neural_act, y_data, HID_DIM, alm_units_start, str_units_start, thal_units_start)
            
        # Save model
        if epoch > 100:
            torch.save(rnn.state_dict(), SAVE_PATH)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss.backward()

        '''
        if MODEL_TYPE == "constraint":
            simple_dynamics_constraint(act, rnn, HID_DIM)
        elif MODEL_TYPE == "no_constraint":
            simple_dynamics_no_constraint(act, rnn, alm_units_start, str_units_start)
        elif MODEL_TYPE == "no_constraint_thal":
            simple_dynamics_constraint(act, rnn, alm_units_start, str_units_start, thal_units_start)
        '''

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
