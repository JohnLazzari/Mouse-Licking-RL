import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sac_model import weights_init_
from torch.distributions import Normal
from thalamocortical_networks import ThalamoCortical
import numpy.random as random

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-6

class Lick_Env(gym.Env):
    def __init__(self, seed, dt, target_time, inp_dim, hid_dim):
        super(Lick_Env, self).__init__()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.dt = dt
        self.target_time = target_time
        self.max_timesteps = int(target_time / dt)
        self.switch = 1
        self.hid_dim = hid_dim
        self.inp_dim = inp_dim

        self.thalamocortical_net = ThalamoCortical(inp_dim, hid_dim)

    def reset(self, episode):
        if episode % 100 == 0:
            if self.switch == 1:
                self.switch = 0.
            else:
                self.switch = 1.
        self.thalamocortical_net.cortical_activity = torch.zeros(size=(self.hid_dim,))
        self.thalamocortical_net.thalamic_activity = torch.zeros(size=(self.inp_dim,))
        state = [*self.thalamocortical_net.thalamic_activity.tolist(), *self.thalamocortical_net.cortical_activity.tolist(), 0., self.switch]
        return state
    
    def _get_reward(self, t, lick):

        # Get target delay time
        if self.switch == 0:
            delay_time = 1
        elif self.switch == 1:
            delay_time = 3

        # Get reward based on the target delay time
        if lick and t >= delay_time:
            return 1 / t 
        
        return 0

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
        with torch.no_grad():
            a_t = torch.tensor(a_t)
            lick_prob = self.thalamocortical_net(a_t)
        
        if random.rand() > lick_prob:
            lick = 1
        else:
            lick = 0
        
        return lick
    
    def _get_next_state(self, lick):
        return [*self.thalamocortical_net.thalamic_activity.tolist(), *self.thalamocortical_net.cortical_activity.tolist(), lick, self.switch]

    def step(self, t, action):
        # Get the lick or no lick
        lick = self._get_lick(action)
        # Get reward
        reward = self._get_reward(t, lick)
        # Get done
        done = self._get_done(t, lick)
        # Get state
        state = self._get_next_state(lick)
        return state, reward, done
   