import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import RNN_MultiRegional
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_delay_data, get_ramp

SAVE_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_gating.pth"
INP_DIM = 1
HID_DIM = 512
OUT_DIM = 1
EPOCHS = 2500
LR = 1e-3
DT = 1e-3
WEIGHT_DECAY = 1e-4

def main():

    ####################################
    #        Training Params           #
    ####################################

    # Create RNN and specifcy objectives
    rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
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
    rnn_optim = optim.Adam(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    ####################################
    #          Train RNN               #
    ####################################

    hn = torch.zeros(size=(1, 3, HID_DIM)).cuda()
    x = torch.zeros(size=(1, 3, HID_DIM)).cuda()

    # mask the losses which correspond to padded values (just in case)
    loss_mask = [torch.ones(size=(length, OUT_DIM), dtype=torch.int) for length in len_seq]
    loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

    loss_mask_act = [torch.ones(size=(length, HID_DIM), dtype=torch.int) for length in len_seq]
    loss_mask_act = pad_sequence(loss_mask_act, batch_first=True).cuda()

    loss_mask_exp = [torch.ones(size=(length, neural_act.shape[-1]), dtype=torch.int) for length in len_seq]
    loss_mask_exp = pad_sequence(loss_mask_exp, batch_first=True).cuda()

    best_loss = np.inf

    str_units_start = 0
    snr_units_start = int(HID_DIM/4)
    thal_units_start = int(HID_DIM/2)
    alm_units_start = int(HID_DIM*(3/4))

    for epoch in range(EPOCHS):
        
        # Pass through RNN
        out, _, act, _, _ = rnn(x_data, hn, x)

        # Get masks
        out = out * loss_mask
        act = act * loss_mask_act
        neural_act = neural_act * loss_mask_exp

        # Get loss
        loss = (1e-3 * criterion(out, y_data) 
                + 1e-2 * constraint_criterion(torch.mean(act[:, :, alm_units_start+(int(HID_DIM/4)-int(0.8*(HID_DIM/4))):], dim=-1, keepdim=True), neural_act)
                + 1e-4 * torch.mean(torch.pow(act, 2), dim=(1, 2, 0))  
                + constraint_criterion(torch.mean(act[:, :, str_units_start:snr_units_start], dim=-1, keepdim=True), neural_act)
                + constraint_criterion(torch.mean(act[:, :, thal_units_start:alm_units_start], dim=-1, keepdim=True), neural_act)
                )
        
        # Save model
        if loss < best_loss and epoch > 100:
            best_loss = loss
            torch.save(rnn.state_dict(), SAVE_PATH)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss.backward()

        # Penalize complex trajectories
        d_act = torch.mean(torch.where(act > 0, 1., 0.), dim=(1, 0))

        '''
        rnn.alm2alm_weight_l0_hh.grad += (1e-4 * rnn.alm2alm_weight_l0_hh.T * d_act[int(HID_DIM*(3/4)):])
        rnn.alm2str_weight_l0_hh.grad += (1e-4 * rnn.alm2str_weight_l0_hh.T * d_act[int(HID_DIM*(3/4)):])
        rnn.thal2str_weight_l0_hh.grad += (1e-4 * rnn.thal2str_weight_l0_hh.T * d_act[int(HID_DIM*(1/2)):int(HID_DIM*(3/4))])
        rnn.str2snr_weight_l0_hh.grad += (1e-4 * rnn.str2snr_weight_l0_hh.T * d_act[:int(HID_DIM/4)])
        rnn.snr2thal_weight_l0_hh.grad += (1e-4 * rnn.snr2thal_weight_l0_hh.T * d_act[int(HID_DIM/4):int(HID_DIM*(1/2))])
        rnn.thal2alm_weight_l0_hh.grad += (1e-4 * rnn.thal2alm_weight_l0_hh.T * d_act[int(HID_DIM*(1/2)):int(HID_DIM*(3/4))])
        '''
        
        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()