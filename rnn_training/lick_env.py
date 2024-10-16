import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import scipy.io as sio
import matplotlib.pyplot as plt

def NormalizeData(data, min, max):
    return (data - min) / (max - min)

class RNN(nn.Module):
    def __init__(self, inp_dim, hid_dim, action_dim, sparse=False):
        super(RNN, self).__init__()

        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        
        self.weight_l0_hh = nn.Parameter(torch.empty(size=(hid_dim, hid_dim)))
        self.weight_l0_ih = nn.Parameter(torch.empty(size=(inp_dim, hid_dim)))
        self.bias_l0_hh = nn.Parameter(torch.empty(size=(hid_dim,)))
        self.bias_l0_ih = nn.Parameter(torch.empty(size=(hid_dim,)))
        nn.init.xavier_uniform_(self.weight_l0_hh)
        nn.init.xavier_uniform_(self.weight_l0_ih)
        nn.init.uniform_(self.bias_l0_hh)
        nn.init.uniform_(self.bias_l0_ih)

        self.fc1 = nn.Linear(hid_dim, action_dim)

    def forward(self, inp: torch.Tensor, hn: torch.Tensor, len_seq=None):

        hn_next = hn.squeeze(0)
        new_hs = []
        for t in range(inp.shape[1]):
            hn_next = torch.sigmoid(hn_next @ self.weight_l0_hh + inp[:, t, :] @ self.weight_l0_ih + self.bias_l0_hh + self.bias_l0_ih)
            new_hs.append(hn_next)
        rnn_out = torch.stack(new_hs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)

        out = torch.sigmoid(self.fc1(rnn_out))
        
        return out, hn_last, rnn_out

#######################################
######## Ramping Environment ##########
#######################################

class Lick_Env_Cont(gym.Env):
    def __init__(self, action_dim, timesteps, thresh, dt, beta, bg_scale, trajectory, full_alm_path, alm_hid_units):

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.thresh = thresh
        self.max_timesteps = timesteps-1
        self.dt = dt
        self.switch_const = 0
        self.cortical_state = 0
        self.switch = 0
        self.cue = 0
        self.beta = beta
        self.bg_scale = bg_scale
        self.num_conds = 1
        self.decay = 0.965
        self.lick = 0
        self.num_stimuli_cue = 100
        self.std_stimuli = 0.1
        self.trajectory = trajectory
        self.alm_activity = {}
        self.alm_hid_units = alm_hid_units
        self.full_alm_path = full_alm_path

        # Load ALM Network
        self.alm_net = RNN(2, self.alm_hid_units, 1)
        checkpoint = torch.load(full_alm_path)
        self.alm_net.load_state_dict(checkpoint)

        # Get the underlying trajectory of the ALM network with a perfect ramp
        # TODO will need to change this when using more conditions, also make it slightly more than just a ramp, have like 10 or 20 timesteps of silence before cue, 
        # lastly, do not overwrite models that perform at least decently well, just have new models be new names that do not overwrite others
        # Create ramp input
        ramp = torch.linspace(0, 1, int((1.1 + (0.6*self.switch)) / self.dt), dtype=torch.float32).unsqueeze(1)
        #baseline = torch.zeros(size=(100, 1))
        #total_ramp = torch.cat((baseline, ramp), dim=0)
        # Create cue input
        cue = torch.zeros_like(ramp)
        cue[0] = 1
        # Concatenate
        total_inp = torch.cat((ramp, cue), dim=-1)
        h0 = torch.zeros(size=(1, 1, self.alm_hid_units))
        # Get psth of activity
        with torch.no_grad():
            _, _, self.target_act = self.alm_net(total_inp.unsqueeze(0), h0)
            self.target_act = self.target_act.squeeze()

    def reset(self, episode: int):

        self.cortical_state = torch.zeros(size=(1, 1, self.alm_hid_units))
        self.cue = 0
        self.cue = 0
        self.lick = 0

        # switch target delay time
        self.switch = episode % self.num_conds
        self.target_delay_time = int((1 + self.switch * 0.6) / self.dt) # scale back since t starts at 0
        self.max_timesteps = self.target_delay_time + 10 # add some extra time so it doesnt have to be exact

        state = [*list(self.cortical_state[0, 0, :]), self.cue, self.switch/self.num_conds]

        return state
    
    def _get_reward(self, t: int):

        reward = 0
        dist = torch.linalg.norm(self.cortical_state.squeeze() - self.target_act[t])
        if self.lick == 1 and t >= self.target_delay_time-1:
            return 5 * ((self.target_delay_time-1) / t)
        if self.lick != 1 and t == self.max_timesteps-1:
            return -5
        if self.lick == 1 and t < self.target_delay_time-1:
            return -5
        if self.trajectory == True:
            if dist > 1:
                return -5
        reward = (1 / (1000**dist))

        return reward
    
    def _get_done(self, t: int):

        done = False
        if t == self.max_timesteps-1:
            done = True
        if self.lick == 1:
            done = True
        if self.trajectory == True:
            dist = torch.linalg.norm(self.cortical_state.squeeze() - self.target_act[t])
            if dist > 1:
                done = True

        return done
    
    def _get_next_state(self, t: int):

        if t == 0:
            self.cue = 1
        else:
            self.cue = 0

        state = [*list(self.cortical_state[0, 0, :]), self.cue, self.switch/self.num_conds]
        return state
    
    def _get_lick(self, action: torch.Tensor):

        action = torch.tensor([action]).unsqueeze(0).unsqueeze(0)
        cue = torch.tensor([self.cue]).unsqueeze(0).unsqueeze(0)
        inp = torch.cat((action, cue), dim=-1)
        with torch.no_grad():
            out, self.cortical_state, _ = self.alm_net(inp, self.cortical_state)
        num = np.random.uniform(0, 1)

        if num <= out.squeeze().item():
            self.lick = 1
        else:
            self.lick = 0
    
    def step(self, t: int, action: torch.Tensor, episodes: int):

        action = action[0]

        self._get_lick(action)

        reward = self._get_reward(t)
        done = self._get_done(t)
        state = self._get_next_state(t)

        return state, reward, done