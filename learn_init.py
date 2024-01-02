import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

INP_DIM = 1
HID_DIM = 8
EPOCHS_LICK = 40_000
EPOCHS_PREP = 1000
LR = 0.001

J_cc = torch.eye(HID_DIM)

class ThalamoCortical_Lick(nn.Module):
    def __init__(self, inp_dim, hid):
        super(ThalamoCortical_Lick, self).__init__()
        self.inp_dim = inp_dim
        self.hid = hid

        # Cortical Weights
        self.J_ct = nn.Parameter(data=torch.randn(hid, inp_dim) / np.sqrt(hid + inp_dim), requires_grad=True)

        # Thalamic Weights
        self.J_tc = nn.Parameter(data=torch.randn(inp_dim, hid) / np.sqrt(hid + inp_dim), requires_grad=True)

        # Thalamic Timescale (not sure what to put)
        self.tau = 1.

        # Readout
        self.W_out = nn.Parameter(torch.randn(hid,) / np.sqrt(HID_DIM), requires_grad=True)

        self.cortical_activity = torch.zeros(size=(hid,)) # will be initialized based on switch time

    def forward(self, x):

        # discrete dynamics with forward euler (dt = 1)
        pre_activity = self.cortical_activity
        self.thalamic_activity = x*self.J_tc @ self.cortical_activity
        self.cortical_activity = J_cc @ self.cortical_activity + self.J_ct @ self.thalamic_activity
        lick_prob = self.W_out @ self.cortical_activity

        return pre_activity, lick_prob

def main():

    targ_lick = torch.linspace(0, 1, int(1/.1))

    criterion = nn.MSELoss()

    lick_net_1s = ThalamoCortical_Lick(INP_DIM, HID_DIM)
    lick_net_3s = ThalamoCortical_Lick(INP_DIM, HID_DIM)

    lick_1s_optimizer = optim.AdamW(lick_net_1s.parameters(), lr=LR)
    lick_3s_optimizer = optim.AdamW(lick_net_3s.parameters(), lr=LR)

    # lick optimization
    for epoch in range(EPOCHS_LICK):

        # meant to just activate a certain unit while silencing others
        lick_net_1s.cortical_activity = torch.ones_like(lick_net_1s.cortical_activity).detach() * 0.1
        lick_inp = torch.tensor([1])

        cortical_series = []
        activity_series = []
        for t in range(targ_lick.shape[0]):
            activity, cortical_out = lick_net_1s(lick_inp)
            cortical_series.append(cortical_out.unsqueeze(0))
            activity_series.append(activity)
        loss = criterion(torch.concatenate(cortical_series), targ_lick) + 0.0001*torch.linalg.norm(torch.stack(activity_series))**2
        print("epoch", epoch, "loss", loss.item())
        lick_1s_optimizer.zero_grad()
        loss.backward()
        lick_1s_optimizer.step()

    # lick optimization
    for epoch in range(EPOCHS_LICK):

        # meant to just activate a certain unit while silencing others
        lick_inp = torch.tensor([1])
        lick_net_3s.cortical_activity = torch.ones_like(lick_net_3s.cortical_activity).detach() * 0.3

        cortical_series = []
        activity_series = []
        for t in range(targ_lick.shape[0]):
            activity, cortical_out = lick_net_3s(lick_inp)
            cortical_series.append(cortical_out.unsqueeze(0))
            activity_series.append(activity)
        loss = criterion(torch.concatenate(cortical_series), targ_lick) + 0.0001*torch.linalg.norm(torch.stack(activity_series))**2
        print("epoch", epoch, "loss", loss.item())
        lick_3s_optimizer.zero_grad()
        loss.backward()
        lick_3s_optimizer.step()

    torch.save({
        'Jct_1s': lick_net_1s.J_ct,
        'Jtc_1s': lick_net_1s.J_tc,
        'Jcc_1s': J_cc,
        'W_out_1s': lick_net_1s.W_out,
        'Jct_3s': lick_net_3s.J_ct,
        'Jtc_3s': lick_net_3s.J_tc,
        'Jcc_3s': J_cc,
        'W_out_3s': lick_net_3s.W_out,
    }, 'checkpoints/thalamocortical_init.pth')


if __name__ == "__main__":
    main()