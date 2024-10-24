import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM
import scipy.io as sio
import matplotlib.pyplot as plt
from utils import gather_inp_data, get_data, get_masks, get_acts_manipulation, get_region_borders
from losses import loss_d1d2, loss_stralm, simple_dynamics_d1d2
from tqdm import tqdm

HID_DIM = 256                                                                       # Hid dim of each region
OUT_DIM = 1451                                                                         # Output dim (not used)
INP_DIM = int(HID_DIM*0.1)                                                          # Input dimension
EPOCHS = 10000                                                                       # Training iterations
LR = 1e-4                                                                           # Learning rate
DT = 1e-2                                                                           # DT to control number of timesteps
WEIGHT_DECAY = 1e-4                                                                 # Weight decay parameter
MODEL_TYPE = "d1d2"                                                                 # d1d2, d1, stralm, d1d2_simple
CONSTRAINED = True                                                                  # Whether or not the model uses plausible circuit
START_SILENCE = 160
END_SILENCE = 220
CONDS = 4
STIM_STRENGTH = 10
EXTRA_STEPS_SILENCE = 100
SILENCED_REGION = "alm"
PCA = False
N_COMPONENTS = 10
INP_TYPE = "simulated"
TRIAL_EPOCH = "full"                                                                                                           # delay or full
INP_PATH = "data/firing_rates/ITIProj_trialPlotAll1.mat"
SAVE_PATH = f"checkpoints/{MODEL_TYPE}_full_simulated_inhibout_256n_noise.1_10000iters.pth"                   # Save path

'''

Default Model(s):
    HID_DIM = 256
    OUT_DIM = 1451
    INP_DIM = int(HID_DIM*0.1)
    EPOCHS = 1000
    LR = 1E-4
    DT = 1E-3
    WEIGHT_DECAY = 1E-3
    MODEL_TYPE = any (d1d2, stralm, d1)
    CONSTRAINED = True

    (no iti input, but data driven)

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

    # Create RNN and specifcy objectives
    if MODEL_TYPE == "d1d2":

        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.1, noise_level_inp=0.0, constrained=CONSTRAINED).cuda()

    elif MODEL_TYPE == "stralm":

        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.01, noise_level_inp=0.01, constrained=CONSTRAINED).cuda()
        
    constraint_criterion = nn.MSELoss()
    thresh_criterion = nn.BCELoss()

    # Get ramping activity
    neural_act, peak_times = get_data(DT, TRIAL_EPOCH, pca=PCA, n_components=N_COMPONENTS)
    neural_act = neural_act.cuda()

    # Get input and output data
    iti_inp, cue_inp, len_seq = gather_inp_data(DT, HID_DIM, INP_PATH, TRIAL_EPOCH, peak_times, inp_type=INP_TYPE)
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Specify Optimizer
    rnn_optim = optim.AdamW(rnn.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    if MODEL_TYPE == "d1d2":

        str_units_start = 0
        alm_start, alm_end = get_region_borders(MODEL_TYPE, "alm_exc", HID_DIM, INP_DIM)

        loss_mask_act = get_masks(OUT_DIM, len_seq)
        inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], rnn.total_num_units), device="cuda")

    elif MODEL_TYPE == "stralm":

        str_units_start = 0
        alm_start, alm_end = get_region_borders(MODEL_TYPE, "alm_exc", HID_DIM, INP_DIM)

        loss_mask_act = get_masks(OUT_DIM, len_seq)
        inhib_stim = torch.zeros(size=(1, iti_inp.shape[1], HID_DIM * 2 + INP_DIM), device="cuda")

    best_steady_state = np.inf
    prev_steady_state = np.inf

    hn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()
    xn = torch.zeros(size=(1, 4, rnn.total_num_units)).cuda()
    
    ###########################
    #    Begin Training       # 
    ###########################
    
    for epoch in range(EPOCHS):
        
        # Pass through RNN
        act, out = rnn(iti_inp, cue_inp, inhib_stim, hn, xn, noise=True)

        # Get masks
        out = out * loss_mask_act

    # Get loss
        if MODEL_TYPE == "d1d2":

            loss = loss_d1d2(
                rnn,
                constraint_criterion, 
                out, 
                neural_act, 
                alm_start,
                alm_end
            )

        elif MODEL_TYPE == "stralm":

            loss = loss_stralm(
                constraint_criterion, 
                thresh_criterion,
                act, 
                neural_act, 
                alm_start, 
                str_units_start, 
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

        '''
        simple_loss = simple_dynamics_d1d2(
            act,
            rnn,
            HID_DIM
        )

        loss += simple_loss
        '''

        # Zero out and compute gradients of above losses
        rnn_optim.zero_grad()
        loss.backward()

        # Take gradient step
        torch.nn.utils.clip_grad_norm_(rnn.parameters(), 1)
        rnn_optim.step()
    
if __name__ == "__main__":
    main()
