import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
import numpy as np
from models import RNN, RNN_Delay
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
    lick_struct[0][194:209] = 1

    # Condition 2
    lick_struct[1] = torch.zeros(size=(240,)).unsqueeze(1)
    lick_struct[1][224:239] = 1

    # Condition 3
    lick_struct[2] = torch.zeros(size=(270,)).unsqueeze(1)
    lick_struct[2][254:269] = 1

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

def gather_population_data(data_folder, region):

    data_struct = {}

    # may need to potentially give the rnn some time varying input as well? (ALM Data)
    for cond in range(3):

        data_struct[cond] = sio.loadmat(f'{data_folder}/{region}_fr_population_cond{cond+1}.mat')['fr_population']
        min_data, max_data = np.min(data_struct[cond]), np.max(data_struct[cond])
        data_struct[cond] = torch.tensor(2 * NormalizeData(np.squeeze(data_struct[cond]), min_data, max_data) - 1, dtype=torch.float32)

    data_total = torch.stack([data_struct[0], data_struct[1], data_struct[2]], dim=0)
    
    return data_total

def main():

    ####################################
    #        Training Params           #
    ####################################

    kinematics_folder = 'data/kinematics'
    save_path = "checkpoints/rnn_goal_delay.pth"
    task = "delay"

    inp_dim = 2
    hid_dim = 2
    out_dim = 1

    # If doing semi data driven semi goal directed
    activity_constraint = False
    region = "striatum"
    data_folder = "data/firing_rates"

    if task == "kinematics":
        rnn_control = RNN(inp_dim, hid_dim, out_dim).cuda()
    elif task == "delay":
        rnn_control = RNN_Delay(inp_dim, hid_dim, out_dim).cuda()

    if task == "kinematics":
        criterion = nn.MSELoss()
    elif task == "delay":
        criterion = nn.BCELoss()
    
    if activity_constraint:
        constraint_criterion = nn.MSELoss()

    epochs = 50_000
    lr = 1e-3

    if task == "kinematics":
        y_data, x_data, len_seq = gather_kinematics_data(kinematics_folder)
    elif task == "delay":
        y_data, x_data, len_seq = gather_delay_data()
    
    if activity_constraint:
        neural_act = gather_population_data(data_folder, region)
    
    rnn_control_optim = optim.AdamW(rnn_control.parameters(), lr=lr, weight_decay=1e-5)

    ####################################
    #          Train RNN               #
    ####################################
    hn = torch.zeros(size=(1, 3, hid_dim), device="cuda")
    # mask the losses which correspond to padded values (just in case)
    loss_mask = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
    loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

    for epoch in range(epochs):
        
        out, _, _ = rnn_control(x_data, hn, len_seq)

        out = out * loss_mask

        loss = criterion(out, y_data)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        rnn_control_optim.zero_grad()
        loss.backward()
        rnn_control_optim.step()
    
    ####################################
    #       Output and Save            #
    ####################################

    with torch.no_grad():

        out, hn, act = rnn_control(x_data, hn, len_seq)
        act = act.cpu().numpy()

        # plot activity for condition 1
        plt.plot(act[0])
        plt.show()

    torch.save(rnn_control.state_dict(), save_path)
    
if __name__ == "__main__":
    main()