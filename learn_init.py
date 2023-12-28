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
EPOCHS = 5000
LR = 0.001

J_cc = torch.randn(HID_DIM, HID_DIM) / np.sqrt(HID_DIM+HID_DIM)
W_out = torch.randn(HID_DIM,) / np.sqrt(HID_DIM)

class ThalamoCortical_Silent(nn.Module):
    def __init__(self, inp_dim, hid):
        super(ThalamoCortical_Silent, self).__init__()
        self.inp_dim = inp_dim
        self.hid = hid

        # Cortical Weights
        self.J_ct = nn.Parameter(data=torch.randn(hid, inp_dim) / np.sqrt(hid + inp_dim), requires_grad=True)

        # Thalamic Weights
        self.J_tc = nn.Parameter(data=torch.randn(inp_dim, hid) / np.sqrt(hid + inp_dim), requires_grad=True)

        # Thalamic Timescale (not sure what to put)
        self.tau = 1.

        self.cortical_activity = torch.zeros(size=(hid,))
        self.thalamic_activity = torch.zeros(size=(inp_dim,))

    def forward(self, x):

        # discrete dynamics with forward euler (dt = 1)
        self.thalamic_activity = self.thalamic_activity - (1/self.tau) * self.thalamic_activity + self.J_tc @ self.cortical_activity + x
        # make this relu just 0 or 1 instead
        self.cortical_activity = J_cc @ self.cortical_activity + self.J_ct @ F.relu(self.thalamic_activity)
        lick_prob = W_out @ self.cortical_activity

        return lick_prob

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
        self.thalamic_activity = torch.zeros(size=(inp_dim,))

    def forward(self, x):

        # discrete dynamics with forward euler (dt = 1)
        self.thalamic_activity = self.thalamic_activity - (1/self.tau) * self.thalamic_activity + self.J_tc @ self.cortical_activity + x
        # make this relu just 0 or 1 instead
        self.cortical_activity = J_cc @ self.cortical_activity + self.J_ct @ F.relu(self.thalamic_activity)
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
        self.thalamic_activity = torch.zeros(size=(inp_dim,))

    def forward(self, x):

        # discrete dynamics with forward euler (dt = 1)
        self.thalamic_activity = self.thalamic_activity - (1/self.tau) * self.thalamic_activity + self.J_tc @ self.cortical_activity + x
        # make this relu just 0 or 1 instead
        self.cortical_activity = J_cc @ self.cortical_activity + self.J_ct @ F.relu(self.thalamic_activity)
        lick_prob = W_out @ self.cortical_activity

        return lick_prob

def main():

    targ_silent = torch.zeros(size=(int(1/.01),))
    targ_lick = torch.linspace(0, 1, int(1/.01))

    # meant to just activate a certain unit while silencing others
    silent_inp = torch.tensor([1])
    lick_inp = torch.tensor([1])
    prep_inp = torch.tensor([1])

    criterion = nn.MSELoss()

    silent_net = ThalamoCortical_Silent(INP_DIM, HID_DIM)
    lick_net = ThalamoCortical_Lick(INP_DIM, HID_DIM)
    prep_net = ThalamoCortical_Prep(INP_DIM, HID_DIM)

    silent_optimizer = optim.Adam(silent_net.parameters(), lr=LR)
    lick_optimizer = optim.Adam(lick_net.parameters(), lr=LR)
    prep_optimizer = optim.Adam(prep_net.parameters(), lr=LR)

    # Silent optimization
    for epoch in range(EPOCHS):

        cortical_series = []
        for t in range(targ_silent.shape[0]):
            cortical_out = silent_net(silent_inp)
            cortical_series.append(cortical_out.unsqueeze(0))
        loss = criterion(torch.concatenate(cortical_series), targ_silent)
        print("epoch", epoch, "loss", loss.item())
        silent_optimizer.zero_grad()
        loss.backward()
        silent_optimizer.step()
        silent_net.cortical_activity = torch.zeros_like(silent_net.cortical_activity)
        silent_net.thalamic_activity = torch.zeros_like(silent_net.thalamic_activity)

    # lick optimization
    for epoch in range(EPOCHS):

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
        lick_net.thalamic_activity = torch.zeros_like(lick_net.thalamic_activity)

    # prep optimization
    # Not doing prep dynamics rn (test what happens without it when switching occurs)
    '''
    for epoch in range(EPOCHS):

        cortical_series = []
        for t in range(targ_lick.shape[0]):
            cortical_out = prep_net(prep_inp)
            cortical_series.append(cortical_out.unsqueeze(0))
        loss = criterion(torch.concatenate(cortical_series), targ_lick)
        print("epoch", epoch, "loss", loss.item())
        prep_optimizer.zero_grad()
        loss.backward()
        prep_optimizer.step()
        prep_net.cortical_activity = torch.zeros_like(prep_net.cortical_activity)
        prep_net.thalamic_activity = torch.zeros_like(prep_net.thalamic_activity)
    '''

    torch.save({
        'silent_Jct': silent_net.J_ct,
        'silent_Jtc': silent_net.J_tc,
        'lick_Jct': lick_net.J_ct,
        'lick_Jtc': lick_net.J_tc,
        'Jcc': J_cc,
        'W_out': W_out,
    }, 'checkpoints/thalamocortical_init.pth')


if __name__ == "__main__":
    main()