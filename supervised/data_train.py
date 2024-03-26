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
    x_inp = {}

    # may need to potentially give the rnn some time varying input as well? (ALM Data)
    for cond in range(3):

        data_struct[cond] = sio.loadmat(f'{data_folder}/{region}_fr_population_cond{cond+1}.mat')['fr_population']
        min_data, max_data = np.min(data_struct[cond]), np.max(data_struct[cond])
        data_struct[cond] = torch.tensor(NormalizeData(np.squeeze(data_struct[cond]), min_data, max_data), dtype=torch.float32)

        x_inp[cond] = torch.tensor([(cond+1)/3], device="cuda", dtype=torch.float32).repeat(300).unsqueeze(1)
        x_inp[cond] = torch.cat([x_inp[cond]], dim=1)

    len_seq = list(map(len, [data_struct[0], data_struct[1], data_struct[2]]))
    x_inp_total = pad_sequence([x_inp[0], x_inp[1], x_inp[2]], batch_first=True).cuda()
    data_total = torch.stack([data_struct[0], data_struct[1], data_struct[2]], dim=0).cuda()
    
    return data_total, x_inp_total, len_seq

def main():

    population_folder = 'data/firing_rates'
    save_path = "checkpoints/rnn_data_striatum.pth"
    region = "striatum"
    sparse = False
    inp_dim = 1
    hid_dim = 2
    out_dim = 533 # 533 for striatum and 517 for alm
    epochs = 100_000
    lr = 1e-4

    rnn_control = RNN(inp_dim, hid_dim, out_dim, sparse=sparse).cuda()

    y_data, x_data, len_seq = gather_population_data(population_folder, region)
    
    rnn_control_optim = optim.AdamW(rnn_control.parameters(), lr=lr)

    criterion = nn.MSELoss()

    ############## Control RNN ######################

    for epoch in range(epochs):
        
        hn = torch.zeros(size=(1, 3, hid_dim), device="cuda")
        out, hn, _ = rnn_control(x_data, hn, len_seq)

        # mask the losses which correspond to padded values (just in case)
        loss_mask = [torch.ones(size=(length, out_dim), dtype=torch.int) for length in len_seq]
        loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

        out = out * loss_mask
        y_data_masked = y_data * loss_mask
        loss = criterion(out, y_data_masked)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        rnn_control_optim.zero_grad()
        loss.backward()
        rnn_control_optim.step()
    
    # Look at output
    with torch.no_grad():

        hn = torch.zeros(size=(1, 3, hid_dim), device="cuda")
        out, hn, act = rnn_control(x_data, hn, len_seq)
        act = act.cpu().numpy()

        # plot activity for condition 1
        plt.plot(act[0])
        plt.show()

    torch.save(rnn_control.state_dict(), save_path)
    
if __name__ == "__main__":
    main()