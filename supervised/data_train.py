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

def gather_population_data(data_folder, region):

    data_struct = {}

    # may need to potentially give the rnn some time varying input as well? (ALM Data)
    for cond in range(3):

        data_struct[cond] = sio.loadmat(f'{data_folder}/{region}_fr_population_cond{cond+1}.mat')['fr_population']
        min_data, max_data = np.min(data_struct[cond]), np.max(data_struct[cond])
        data_struct[cond] = torch.tensor(NormalizeData(np.squeeze(data_struct[cond]), min_data, max_data), dtype=torch.float32)

    len_seq = [300, 300, 300]
    data_total = torch.stack([data_struct[0], data_struct[1], data_struct[2]], dim=0).cuda()
    
    return data_total, len_seq

def gather_inp_data(psth_folder, region):
    
    data_struct = {}
    if region == "alm":
        inp_region = "striatum"
    elif region == "striatum":
        inp_region = "alm"

    for cond in range(3):
        data_struct[cond] = sio.loadmat(f'{psth_folder}/{inp_region}_PSTH_cond{cond+1}.mat')['psth']
        min_data, max_data = np.min(data_struct[cond]), np.max(data_struct[cond])
        data_struct[cond] = torch.tensor(NormalizeData(np.squeeze(data_struct[cond]), min_data, max_data), dtype=torch.float32).unsqueeze(-1)

    cue_inp = torch.zeros(size=(3, 300, 1))
    cue_inp[:, 99, :] = 1
    ramp_inp_total = torch.stack([data_struct[0], data_struct[1], data_struct[2]], dim=0)
    total_inp = torch.cat((ramp_inp_total, cue_inp), dim=-1).cuda()

    return total_inp

def main():

    population_folder = 'data/firing_rates'
    psth_folder = 'data/PCs_PSTH'
    save_path = "checkpoints/rnn_data_alm.pth"
    region = "alm"
    inp_dim = 2
    hid_dim = 2
    out_dim = 517 # 533 for striatum and 517 for alm
    epochs = 100_000
    lr = 1e-3

    rnn_control = RNN(inp_dim, hid_dim, out_dim).cuda()

    y_data, len_seq = gather_population_data(population_folder, region)
    x_data = gather_inp_data(psth_folder, region)
    
    rnn_control_optim = optim.AdamW(rnn_control.parameters(), lr=lr, weight_decay=1e-3)

    criterion = nn.MSELoss()

    ############## Control RNN ######################
    hn = torch.zeros(size=(1, 3, hid_dim), device="cuda")
    # mask the losses which correspond to padded values (just in case)
    loss_mask = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
    loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

    for epoch in range(epochs):
        
        out, _, _ = rnn_control(x_data, hn)

        loss = criterion(out, y_data)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        rnn_control_optim.zero_grad()
        loss.backward()
        rnn_control_optim.step()
    
    torch.save(rnn_control.state_dict(), save_path)
    
if __name__ == "__main__":
    main()