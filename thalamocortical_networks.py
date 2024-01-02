import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

class ThalamoCortical(nn.Module):
    def __init__(self, inp_dim, hid):
        super(ThalamoCortical, self).__init__()
        self.inp_dim = inp_dim
        self.hid = hid

        # Unload fixed weights
        weights = torch.load("checkpoints/thalamocortical_init.pth")

        # Cortical Weights
        self.J_cc_1s = weights["Jcc_1s"]
        self.J_ct_1s = weights["Jct_1s"]

        self.J_cc_3s = weights["Jcc_3s"]
        self.J_ct_3s = weights["Jct_3s"]

        # Thalamic Weights
        self.J_tc_1s = weights["Jtc_1s"]
        self.J_tc_3s = weights["Jtc_3s"]

        # Thalamic Timescale (not sure what to put)
        self.tau = 1.

        # Readout for probability
        self.W_out_1s = weights["W_out_1s"]
        self.W_out_3s = weights["W_out_3s"]

        self.cortical_activity = torch.zeros(size=(hid,))

        self.prev_action = torch.tensor([0])

    ''' TODO try out 3 different environments
    1. sustained activity without thalamocortical dynamics, state is only simple ramping value, lick or no lick, and switch
    2. sustained activity with thalamocortical dynamics, state contains the thalamocortical activity with learned values from ramping (current)
    3. sustained activity with thalamocortical dynamics, the feedback is the actual ALM activity and if sustained then a lick occurs
    (since the output of the BG is always 1 or 0 and holding that for a certain period of time, having an actual cortical model may not be necessary)
    (However, if having actual dynamics in the feedback helps then that could be useful, but actually training a model may be unnessecary)
    (maybe also add a cue to the environment and start from trial onset with cue 1s after trial onset)
    (Can learn the delay with a single target time, therefore only try the above with switching)
    (Its also possible that each of the above feedback mechanisms can be implemented with separate dynamics for each target time, which may help it differentiate)
    '''
    def forward(self, x, switch):

        if not torch.equal(x, self.prev_action) and switch == 0:
            self.cortical_activity = torch.ones(size=(self.hid,)) * 0.1
        elif not torch.equal(x, self.prev_action) and switch == 1:
            self.cortical_activity = torch.ones(size=(self.hid,)) * 0.3
        self.prev_action = x

        # discrete dynamics with forward euler (dt = 1)
        if switch == 0:
            self.thalamic_activity = x * self.J_tc_1s @ self.cortical_activity
            self.cortical_activity = self.J_cc_1s @ self.cortical_activity + self.J_ct_1s @ self.thalamic_activity
            lick_prob = self.W_out_1s @ self.cortical_activity
        else:
            self.thalamic_activity = x * self.J_tc_3s @ self.cortical_activity
            self.cortical_activity = self.J_cc_3s @ self.cortical_activity + self.J_ct_3s @ self.thalamic_activity
            lick_prob = self.W_out_3s @ self.cortical_activity

        return lick_prob
