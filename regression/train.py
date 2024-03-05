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
from custom_optim import CustomAdamOptimizer

def NormalizeData(data, min, max):
    return (data - min) / (max - min)

def main():

    kinematics_folder = 'data/kinematics'
    alm_data_path = "data/PCs_PSTH"
    sparse = False

    kinematics_jaw_x = {}
    kinematics_jaw_y = {}
    alm_pcs = {}
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

        # ALM Data
        alm_timepoints = np.linspace(-1, 2, 300).round(2)
        kin_timepoints = Taxis[cond].round(2)
        # Get a mask
        time_mask = []
        kin_idx = 0
        for timepoint in alm_timepoints:
            if timepoint == kin_timepoints[kin_idx] and kin_idx < kin_timepoints.shape[0]:
                time_mask.append(True)
                if kin_idx < kin_timepoints.shape[0]-1:
                    kin_idx += 1
            else:
                time_mask.append(False)

        alm_pcs[cond] = sio.loadmat(f'{alm_data_path}/alm_5PCs_cond{cond+1}.mat')['projected_data']
        alm_pcs[cond] = NormalizeData(np.squeeze(alm_pcs[cond]), np.min(alm_pcs[cond]), np.max(alm_pcs[cond]))
        alm_pcs[cond] = torch.tensor(alm_pcs[cond][time_mask], device="cuda", dtype=torch.float32).unsqueeze(2)

        x_inp[cond] = torch.tensor([(cond+1)/3], device="cuda", dtype=torch.float32).repeat(kinematics_jaw_y[cond].shape[0]).unsqueeze(1)
        x_inp[cond] = torch.cat([x_inp[cond], alm_pcs[cond][:, 0], alm_pcs[cond][:, 1], alm_pcs[cond][:, 2], alm_pcs[cond][:, 3], alm_pcs[cond][:, 4]], dim=1)

    rnn_control = RNN(6, 256, 2, sparse=sparse).cuda()
    criterion = nn.MSELoss()
    epochs = 15_000
    lr = 1e-3
    
    rnn_control_optim = optim.AdamW(rnn_control.parameters(), lr=lr, weight_decay=1e-3)

    ############## Control RNN ######################

    for epoch in range(epochs):
        
        hn = torch.zeros(size=(1, 3, 256), device="cuda")

        len_seq = list(map(len, [kinematics_jaw_x[0], kinematics_jaw_x[1], kinematics_jaw_x[2]]))
        x_inp_total = pad_sequence([x_inp[0], x_inp[1], x_inp[2]], batch_first=True).cuda()
        kinematics_x_total = pad_sequence([kinematics_jaw_x[0], kinematics_jaw_x[1], kinematics_jaw_x[2]], batch_first=True).cuda()
        kinematics_y_total = pad_sequence([kinematics_jaw_y[0], kinematics_jaw_y[1], kinematics_jaw_y[2]], batch_first=True).cuda()
        kinematics_total = torch.cat((kinematics_x_total, kinematics_y_total), dim=-1).cuda()

        out, hn, activity = rnn_control(x_inp_total, hn, len_seq)

        # mask the losses which correspond to padded values (just in case)
        loss_mask = [torch.ones(size=(length, 2), dtype=torch.int) for length in len_seq]
        loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

        out *= loss_mask
        kinematics_total *= loss_mask
        loss = criterion(out, kinematics_total)

        print("Training loss at epoch {}:{}".format(epoch, loss.item()))

        rnn_control_optim.zero_grad()
        loss.backward()
        rnn_control_optim.step()
    
    # Look at output
    with torch.no_grad():

        hn = torch.zeros(size=(1, 3, 256), device="cuda")
        out, hn, act = rnn_control(x_inp_total, hn, len_seq)
        act = act.cpu().numpy()

        out_x = []
        out_x.append(out[0, :len_seq[0], 0])
        out_x.append(out[1, :len_seq[1], 0])
        out_x.append(out[2, :len_seq[2], 0])

        out_y = []
        out_y.append(out[0, :len_seq[0], 1])
        out_y.append(out[1, :len_seq[1], 1])
        out_y.append(out[2, :len_seq[2], 1])

        plt.plot(out_x[0].cpu().numpy())
        plt.plot(out_x[1].cpu().numpy())
        plt.plot(out_x[2].cpu().numpy())
        plt.show()

        plt.plot(out_y[0].cpu().numpy())
        plt.plot(out_y[1].cpu().numpy())
        plt.plot(out_y[2].cpu().numpy())
        plt.show()

    np.save("results/rnn_control_regression_act.npy", act)
    
if __name__ == "__main__":
    main()