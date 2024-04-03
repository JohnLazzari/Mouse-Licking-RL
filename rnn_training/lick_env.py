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
        nn.init.xavier_uniform_(self.weight_l0_hh)
        nn.init.xavier_uniform_(self.weight_l0_ih)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, action_dim)

    def forward(self, inp: torch.Tensor, hn: torch.Tensor, len_seq=None):

        hn_next = hn.squeeze(0)
        new_hs = []
        for t in range(inp.shape[1]):
            hn_next = torch.sigmoid(hn_next @ self.weight_l0_hh + inp[:, t, :] @ self.weight_l0_ih)
            new_hs.append(hn_next)
        rnn_out = torch.stack(new_hs, dim=1)
        hn_last = rnn_out[:, -1, :].unsqueeze(0)

        out = F.relu(self.fc1(rnn_out))
        out = torch.sigmoid(self.fc2(out))
        
        return out, hn_last, rnn_out

#######################################
######## Ramping Environment ##########
#######################################

class Lick_Env_Cont(gym.Env):
    def __init__(self, action_dim, timesteps, thresh, dt, beta, bg_scale, alm_data_path):

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
        self.alm_data_path = alm_data_path
        self.num_conds = 1
        self.decay = 0.965
        self.lick = 0
        self.num_stimuli_cue = 100
        self.std_stimuli = 0.1
        self.alm_activity = {}

        # Load ALM Network
        self.alm_net = RNN(2, 32, 1)
        checkpoint = torch.load("checkpoints/rnn_goal_data_full_delay.pth")
        self.alm_net.load_state_dict(checkpoint)
    
    def reset(self, episode: int):

        self.cortical_state = torch.zeros(size=(1, 1, 32))
        self.cue = 0
        self.lick = 0

        # switch target delay time
        self.switch = episode % self.num_conds
        self.target_delay_time = int((2 + self.switch * 0.6) / self.dt) # scale back since t starts at 0
        self.max_timesteps = self.target_delay_time + 20 # add some extra time so it doesnt have to be exact

        state = [*list(self.cortical_state[0, 0, :]), self.cue, (self.switch+1)/(self.num_conds+1)]

        return state
    
    def _get_reward(self, t: int):

        reward = 0
        if self.lick == 1 and t >= self.target_delay_time-1:
            reward += ((self.target_delay_time-1) / t)
        if self.lick != 1 and t == self.max_timesteps-1:
            reward -= 1
        if self.lick == 1 and t < self.target_delay_time-1:
            reward -= 1

        return reward
    
    def _get_done(self, t: int):

        done = False
        if t == self.max_timesteps-1:
            done = True
        if self.lick == 1:
            done = True
        return done
    
    def _get_next_state(self, t: int):

        if t == 99:
            self.cue = 1
        else:
            self.cue = 0

        state = [*list(self.cortical_state[0, 0, :]), self.cue, (self.switch+1)/(self.num_conds+1)]
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

    
#######################################
##### Kinematics Jaw Environment ######
#######################################
    
class Kinematics_Jaw_Env(gym.Env):
    def __init__(self, action_dim, dt, kinematics_folder, alm_data_path, bg_scale):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_dim = action_dim
        self.dt = dt
        self.cue = 0
        self.cue_time = 1 / dt
        self.bg_scale = bg_scale
        self.kinematics_folder = kinematics_folder
        self.alm_data_path = alm_data_path
        self.thresh = 0.1
        self.fixed_steps = 1
        self.max_timesteps = None
        self.cur_cond = None
        self.cortical_state = np.zeros(shape=(action_dim,))
        self.kinematics_jaw_x = {}
        self.kinematics_jaw_y = {}
        self.alm_pcs = {}
        self.Taxis = {}

        # Load data
        for cond in range(3):

            # Kinematic Data
            self.kinematics_jaw_y[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}y_jaw.mat')['condy_jaw_mean']
            self.kinematics_jaw_x[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}x_jaw.mat')['condx_jaw_mean']
            # y position is lower than x position, using these min and max values such that the scaling between x and y is accurate
            min_jaw_y, max_jaw_y = np.min(self.kinematics_jaw_y[cond]), np.max(self.kinematics_jaw_y[cond])
            y_diff = max_jaw_y - min_jaw_y
            # we want to have them be between 0 and 1 but at a reasonable scale
            min_jaw_x, max_jaw_x = np.min(self.kinematics_jaw_x[cond]), np.min(self.kinematics_jaw_x[cond]) + y_diff

            self.kinematics_jaw_y[cond] = NormalizeData(np.squeeze(self.kinematics_jaw_y[cond]), min_jaw_y, max_jaw_y)
            self.kinematics_jaw_x[cond] = NormalizeData(np.squeeze(self.kinematics_jaw_x[cond]), min_jaw_x, max_jaw_x)

            self.Taxis[cond] = sio.loadmat(f'{kinematics_folder}/Taxis_cond{cond+1}.mat')['Taxis_cur'].squeeze()
            
            # ALM Data
            alm_timepoints = np.linspace(-1, 2, 300).round(2)
            kin_timepoints = self.Taxis[cond].round(2)
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

            self.alm_pcs[cond] = sio.loadmat(f'{alm_data_path}/alm_5PCs_cond{cond+1}.mat')['projected_data']
            self.alm_pcs[cond] = NormalizeData(np.squeeze(self.alm_pcs[cond]), np.min(self.alm_pcs[cond]), np.max(self.alm_pcs[cond]))
            self.alm_pcs[cond] = self.alm_pcs[cond][time_mask]

    def reset(self, episode: int):

        # Get current kinematic condition
        self.cur_cond = episode % 3
        
        # Assert the shapes are the same for x and y vals
        assert self.kinematics_jaw_x[self.cur_cond].shape == self.kinematics_jaw_y[self.cur_cond].shape

        # Gather environment variables for episode
        self.max_timesteps = self.kinematics_jaw_x[self.cur_cond].shape[0]
        self.speed_const = (self.cur_cond + 1) / 3
        self.cue = 0
        self.thresh = 0.1
        self.cortical_state = np.zeros(shape=(self.action_dim,))

        # [pred_x_pos, pred_y_pos, true_x_pos, true_y_pos, speed_const, cue]
        state = [0., 
                0., 
                self.speed_const, 
                self.cue]

        return state
    
    def _get_reward(self, t: int):

        dist_x_jaw = abs(self.cortical_state[0] - self.kinematics_jaw_x[self.cur_cond][t])
        dist_y_jaw = abs(self.cortical_state[1] - self.kinematics_jaw_y[self.cur_cond][t])

        if dist_x_jaw > self.thresh or dist_y_jaw > self.thresh:
            reward = -5
            return reward

        reward_x_jaw = (1 / 1000**(dist_x_jaw))
        reward_y_jaw = (1 / 1000**(dist_y_jaw))

        reward = reward_x_jaw + reward_y_jaw

        # add reward based on lick
        if self.cur_cond == 0:
            if 0.95 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 1.1:
                reward += 5
        elif self.cur_cond == 1:
            if 1.25 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 1.4:
                reward += 5
        elif self.cur_cond == 2:
            if 1.55 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 1.7:
                reward += 5

        return reward
    
    def _get_done(self, t: int):

        done = False

        dist_x_jaw = abs(self.cortical_state[0] - self.kinematics_jaw_x[self.cur_cond][t])
        dist_y_jaw = abs(self.cortical_state[1] - self.kinematics_jaw_y[self.cur_cond][t])

        if dist_x_jaw > self.thresh or dist_y_jaw > self.thresh:
            done = True
        if t == self.max_timesteps-1:
            done = True
        return done
    
    def _get_next_state(self, t: int):

        # change cue based on Taxis
        if -0.033 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 0.033:
            self.cue = 1

        # change cue based on condition
        if self.cur_cond == 0:
            if 0.95 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 1.1:
                self.cue = 0
        elif self.cur_cond == 1:
            if 1.25 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 1.4:
                self.cue = 0
        elif self.cur_cond == 2:
            if 1.55 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 1.7:
                self.cue = 0

        state = [self.cortical_state[0], 
                self.cortical_state[1], 
                self.speed_const, 
                self.cue]

        return state
    
    def _get_pred_kinematics(self, action):
        action = np.array(action)
        # Use simple linear dynamics for now
        self.cortical_state = self.cortical_state + action * self.bg_scale

    def step(self, t: int, action: torch.Tensor, episode_num: int):

        self._get_pred_kinematics(action)
        reward = self._get_reward(t)
        done = self._get_done(t)
        state = self._get_next_state(t)
        return state, reward, done
    