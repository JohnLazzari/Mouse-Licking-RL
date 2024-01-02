import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from thalamocortical_networks import ThalamoCortical
import numpy.random as random

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-6

class Lick_Env(gym.Env):
    def __init__(self, seed, dt, target_time, inp_dim, hid_dim, mode):
        super(Lick_Env, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.dt = dt
        self.target_time = target_time
        self.max_timesteps = int(target_time / dt)
        self.mode = mode
        self.switch = 1
        self.hid_dim = hid_dim
        self.inp_dim = inp_dim
        self.ramp = 0 # for no feedback dynamics in environment

        self.thalamocortical_net = ThalamoCortical(inp_dim, hid_dim)

    def reset(self, episode):

        if episode % 100 == 0:

            if self.switch == 1:
                self.switch = 0.
            else:
                self.switch = 1.

        if self.mode == "learned_dynamics":

            if self.switch == 0:
                self.thalamocortical_net.cortical_activity = torch.ones(size=(self.hid_dim,)) * 0.1
            elif self.switch == 1:
                self.thalamocortical_net.cortical_activity = torch.ones(size=(self.hid_dim,)) * 0.3

            self.thalamocortical_net.thalamic_activity = torch.zeros(size=(self.inp_dim,))
            self.thalamocortical_net.prev_action = torch.zeros_like(self.thalamocortical_net.prev_action)

            state = [*self.thalamocortical_net.thalamic_activity.tolist(), *self.thalamocortical_net.cortical_activity.tolist(), 0., self.switch, 0.]

        elif self.mode == "no_dynamics":

            self.ramp = 0.
            state = [self.ramp, 0., self.switch, 0.]

        return state
    
    def _get_reward(self, t, lick, action):

        reward = 0

        # Get target delay time (t starts at zero)
        if self.switch == 0:
            delay_time = int(2/self.dt) - 1
        elif self.switch == 1:
            delay_time = int(4/self.dt) - 1

        # Get reward based on the target delay time
        if lick and t >= delay_time:
            reward += (delay_time / t)
        
        return reward

    def _get_done(self, t, lick):

        # If lick then done
        if lick:
            return 1
        
        # If last timestep then done
        if t == self.max_timesteps-1:
            return 1
        
        return 0
    
    def _get_lick(self, a_t):
        # Get the lick depending on probability
        if self.mode == "learned_dynamics":
            with torch.no_grad():
                a_t = torch.tensor(a_t)
                ramp = self.thalamocortical_net(a_t, self.switch)
            if ramp >= .99:
                lick = 1
            else:
                lick = 0
        elif self.mode == "no_dynamics":
            if a_t[0] == 1:
                self.ramp += (a_t[0] * 0.1)
            else:
                self.ramp = 0
            if self.ramp >= .99:
                lick = 1
            else:
                lick = 0

        return lick
    
    def _get_next_state(self, lick, t):
        # Thalamocortical network gives feedback to basal ganglia
        if self.mode == "learned_dynamics":
            state =  [*self.thalamocortical_net.thalamic_activity.tolist(), *self.thalamocortical_net.cortical_activity.tolist(), lick, self.switch, (t*self.dt) / (self.target_time)]
        elif self.mode == "no_dynamics":
            state =  [self.ramp, lick, self.switch, (t*self.dt) / (self.target_time)]
        return state

    def step(self, t, action):
        # Get the lick or no lick
        lick = self._get_lick(action)
        # Get reward
        reward = self._get_reward(t, lick, action)
        # Get done
        done = self._get_done(t, lick)
        # Get state
        state = self._get_next_state(lick, t)
        return state, reward, done
   