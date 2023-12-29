import torch
import torch.nn as nn
import torch.optim as optim
from sac_model import Actor
import scipy.io
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

INP_DIM = 1
HID_DIM = 64
EPOCHS_LICK = 2500
EPOCHS_PREP = 1000
LR = 0.001

J_cc = torch.randn(HID_DIM, HID_DIM) / np.sqrt(HID_DIM+HID_DIM)
W_out = torch.randn(HID_DIM,) / np.sqrt(HID_DIM)


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

        self.cortical_activity = torch.zeros(size=(hid,))

    def forward(self, x):

        # discrete dynamics with forward euler (dt = 1)
        self.thalamic_activity = self.J_tc @ self.cortical_activity + x
        self.cortical_activity = J_cc @ self.cortical_activity + self.J_ct @ self.thalamic_activity
        lick_prob = W_out @ self.cortical_activity

        return lick_prob

class ThalamoCortical_Prep(nn.Module):
    def __init__(self, inp_dim, hid):
        super(ThalamoCortical_Prep, self).__init__()
        self.inp_dim = inp_dim
        self.hid = hid

        # Cortical Weights
        self.J_ct = nn.Parameter(data=torch.randn(hid, inp_dim) / np.sqrt(hid + inp_dim), requires_grad=True)

        # Thalamic Weights
        self.J_tc = nn.Parameter(data=torch.randn(inp_dim, hid) / np.sqrt(hid + inp_dim), requires_grad=True)

        # Thalamic Timescale (not sure what to put)
        self.tau = 1.

        self.cortical_activity = torch.zeros(size=(hid,))
        self.c_init_mu = torch.zeros(size=(hid,))

    def forward(self):

        # discrete dynamics with forward euler (dt = 1)
        x_mu = - J_cc @ self.c_init_mu + self.J_ct @ self.J_tc @ self.c_init_mu + torch.eye(HID_DIM) @ self.c_init_mu
        self.cortical_activity = torch.eye(HID_DIM) @ self.cortical_activity + J_cc @ self.cortical_activity + self.J_ct @ self.J_tc @ self.cortical_activity - torch.eye(HID_DIM) @ self.cortical_activity + x_mu
        lick = W_out @ self.cortical_activity

        return self.cortical_activity, lick

def main():

    targ_lick = torch.linspace(0, 1, int(1/.01))**2

    # meant to just activate a certain unit while silencing others
    lick_inp = torch.tensor([1])

    criterion = nn.MSELoss()

    lick_net = ThalamoCortical_Lick(INP_DIM, HID_DIM)
    prep_net = ThalamoCortical_Prep(8, HID_DIM)

    lick_optimizer = optim.AdamW(lick_net.parameters(), lr=LR, weight_decay=0.1)
    prep_optimizer = optim.AdamW(prep_net.parameters(), lr=LR, weight_decay=0.1)

    # lick optimization
    for epoch in range(EPOCHS_LICK):

        cortical_series = []
        for t in range(targ_lick.shape[0]):
            cortical_out = lick_net(lick_inp)
            cortical_series.append(cortical_out.unsqueeze(0))
        loss = criterion(torch.concatenate(cortical_series), targ_lick)
        print("epoch", epoch, "loss", loss.item())
        lick_optimizer.zero_grad()
        loss.backward()
        lick_optimizer.step()
        lick_net.cortical_activity = torch.zeros_like(lick_net.cortical_activity)

    '''
    # prep optimization
    # Not doing prep dynamics rn (test what happens without it when switching occurs)
    for epoch in range(EPOCHS_PREP):

        cortical_series = []
        lick_series = []
        prep_net.cortical_activity = torch.FloatTensor(HID_DIM).uniform_(-15, 15)
        for t in range(25):
            corical_out, out = prep_net()
            cortical_series.append(corical_out.unsqueeze(0))
            lick_series.append(out.unsqueeze(0))
        loss = criterion(torch.concatenate(cortical_series)[-1], prep_net.c_init_mu) + .001 * torch.sum(torch.concatenate(lick_series)**2)
        print("epoch", epoch, "loss", loss.item())
        prep_optimizer.zero_grad()
        loss.backward()
        prep_optimizer.step()
    '''

    torch.save({
        'lick_Jct': lick_net.J_ct,
        'lick_Jtc': lick_net.J_tc,
        'Jcc': J_cc,
        'W_out': W_out,
    }, 'checkpoints/thalamocortical_init.pth')


if __name__ == "__main__":
    main()