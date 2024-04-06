import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN
import scipy.io as sio
import matplotlib.pyplot as plt

def NormalizeData(data, min, max):
    return (data - min) / (max - min)

def gather_kinematics_data(kinematics_folder):

    kinematics_jaw_x = {}
    kinematics_jaw_y = {}
    Taxis = {}
    x_inp = {}

    # may need to potentially give the rnn some time varying input as well? (ALM Data)
    for cond in range(3):

        kinematics_jaw_y[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}y_jaw.mat')['condy_jaw_mean']
        kinematics_jaw_x[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}x_jaw.mat')['condx_jaw_mean']
        # y position is lower than x position, using these min and max values such that the scaling between x and y is accurate
        min_jaw_y, max_jaw_y = np.min(kinematics_jaw_y[cond]), np.max(kinematics_jaw_y[cond])
        y_diff = max_jaw_y - min_jaw_y
        # we want to have them be between 0 and 1 but at a reasonable scale
        min_jaw_x, max_jaw_x = np.min(kinematics_jaw_x[cond]), np.min(kinematics_jaw_x[cond]) + y_diff

        kinematics_jaw_y[cond] = torch.tensor(NormalizeData(np.squeeze(kinematics_jaw_y[cond]), min_jaw_y, max_jaw_y), dtype=torch.float32).unsqueeze(1)
        kinematics_jaw_x[cond] = torch.tensor(NormalizeData(np.squeeze(kinematics_jaw_x[cond]), min_jaw_x, max_jaw_x), dtype=torch.float32).unsqueeze(1)

        Taxis[cond] = sio.loadmat(f'{kinematics_folder}/Taxis_cond{cond+1}.mat')['Taxis_cur'].squeeze()

        x_inp[cond] = torch.tensor([(cond+1)/3], device="cuda", dtype=torch.float32).repeat(kinematics_jaw_y[cond].shape[0]).unsqueeze(1)
        x_inp[cond] = torch.cat([x_inp[cond]], dim=1)

    len_seq = list(map(len, [kinematics_jaw_x[0], kinematics_jaw_x[1], kinematics_jaw_x[2]]))
    x_inp_total = pad_sequence([x_inp[0], x_inp[1], x_inp[2]], batch_first=True).cuda()
    kinematics_x_total = pad_sequence([kinematics_jaw_x[0], kinematics_jaw_x[1], kinematics_jaw_x[2]], batch_first=True).cuda()
    kinematics_y_total = pad_sequence([kinematics_jaw_y[0], kinematics_jaw_y[1], kinematics_jaw_y[2]], batch_first=True).cuda()
    kinematics_total = torch.cat((kinematics_x_total, kinematics_y_total), dim=-1).cuda()
    
    return kinematics_total, x_inp_total, len_seq

def gather_delay_data():
    
    lick_struct = {}
    ramp_inp = {}

    # Condition 1
    lick_struct[0] = torch.zeros(size=(210,)).unsqueeze(1)
    lick_struct[0][200:210] = 1

    # Condition 2
    lick_struct[1] = torch.zeros(size=(240,)).unsqueeze(1)
    lick_struct[1][230:240] = 1

    # Condition 3
    lick_struct[2] = torch.zeros(size=(270,)).unsqueeze(1)
    lick_struct[2][260:270] = 1

    for cond in range(3):
        ramp = torch.linspace(0, 1, int((1.1 + (.3*cond)) / 0.01), dtype=torch.float32).unsqueeze(1)
        baseline = torch.zeros(size=(100, 1))
        ramp_inp[cond] = torch.cat((baseline, ramp), dim=0)

    len_seq = [210, 240, 270]
    cue_inp = torch.zeros(size=(3, 270, 1))
    cue_inp[:, 99, :] = 1
    ramp_inp_total = pad_sequence([ramp_inp[0], ramp_inp[1], ramp_inp[2]], batch_first=True)
    total_inp = torch.cat((ramp_inp_total, cue_inp), dim=-1).cuda()
    lick_seq_total = pad_sequence([lick_struct[0], lick_struct[1], lick_struct[2]], batch_first=True).cuda()

    return lick_seq_total, total_inp, len_seq

def gather_population_data(data_folder, region, linear=True):

    data_struct = {}
    lens = [210, 240, 270]

    for cond in range(3):

        if linear:
            ramp = torch.linspace(0, 1, int((1.1 + (.3*cond)) / 0.01), dtype=torch.float32).unsqueeze(1)
            baseline = torch.zeros(size=(100, 1))
            data_struct[cond] = torch.cat((baseline, ramp), dim=0)
        else:
            data_struct[cond] = sio.loadmat(f'{data_folder}/{region}_fr_population_cond{cond+1}.mat')['fr_population']
            min_data, max_data = np.min(data_struct[cond]), np.max(data_struct[cond])
            data_struct[cond] = torch.tensor(NormalizeData(np.squeeze(data_struct[cond]), min_data, max_data), dtype=torch.float32)[:lens[cond]]

    data_total = pad_sequence([data_struct[0], data_struct[1], data_struct[2]], batch_first=True).cuda()
    
    return data_total

def main():

    ####################################
    #        Training Params           #
    ####################################

    kinematics_folder = 'data/kinematics'
    save_path = "checkpoints/rnn_goal_data_full_delay.pth"
    task = "delay"

    inp_dim = 2
    hid_dim = 517
    out_dim = 1

    # If doing semi data driven semi goal directed
    activity_constraint = True
    linear = False
    region = "alm"
    data_folder = "data/firing_rates"

    rnn_control = RNN(inp_dim, hid_dim, out_dim).cuda()

    if task == "kinematics":
        criterion = nn.MSELoss()
    elif task == "delay":
        criterion = nn.BCELoss()
    
    if activity_constraint:
        constraint_criterion = nn.MSELoss()

    epochs = 100_000
    lr = 1e-3

    if task == "kinematics":
        y_data, x_data, len_seq = gather_kinematics_data(kinematics_folder)
    elif task == "delay":
        y_data, x_data, len_seq = gather_delay_data()
    
    if activity_constraint:
        neural_act = gather_population_data(data_folder, region, linear=linear)
    
    plt.plot(neural_act[0, :len_seq[0], :].cpu().numpy())
    plt.show()
    
    rnn_control_optim = optim.AdamW(rnn_control.parameters(), lr=lr, weight_decay=1e-3)

    ####################################
    #          Train RNN               #
    ####################################

    hn = torch.zeros(size=(1, 3, hid_dim)).cuda()

    # mask the losses which correspond to padded values (just in case)
    loss_mask = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
    loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

    loss_mask_act = [torch.ones(size=(length, hid_dim), dtype=torch.int) for length in len_seq]
    loss_mask_act = pad_sequence(loss_mask_act, batch_first=True).cuda()

    loss_mask_exp = [torch.ones(size=(length, neural_act.shape[-1]), dtype=torch.int) for length in len_seq]
    loss_mask_exp = pad_sequence(loss_mask_exp, batch_first=True).cuda()

    best_loss = np.inf

    for epoch in range(epochs):
        
        out, _, act = rnn_control(x_data, hn)

        out = out * loss_mask

        if activity_constraint and linear:
            act = act * loss_mask_act
            neural_act = neural_act * loss_mask_exp
            loss = 1e-3 * criterion(out, y_data) + constraint_criterion(torch.mean(act, dim=-1, keepdim=True), neural_act) + 1e-4 * torch.mean(torch.pow(act, 2), dim=(1, 2, 0))
        elif activity_constraint and not linear:
            act = act * loss_mask_act
            neural_act = neural_act * loss_mask_exp
            loss = 1e-3 * criterion(out, y_data) + constraint_criterion(act, neural_act) + 1e-4 * torch.mean(torch.pow(act, 2), dim=(1, 2, 0))
        else:
            loss = criterion(out, y_data)
        
        if loss < best_loss and epoch > 5000:
            best_loss = loss
            torch.save(rnn_control.state_dict(), save_path)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        # Zero out and compute gradients of above losses
        rnn_control_optim.zero_grad()
        loss.backward()

        # Implement gradient of complicated trajectory loss
        d_act = torch.mean(torch.pow(act * (1 - act), 2), dim=(1, 0))
        rnn_control.weight_l0_hh.grad += (1e-4 * rnn_control.weight_l0_hh * d_act)

        # Take gradient step
        rnn_control_optim.step()
    
if __name__ == "__main__":
    main()