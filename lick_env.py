import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sac_model import weights_init_
from torch.distributions import Normal
from alm_networks import ALM

LOG_SIG_MIN = -20
LOG_SIG_MAX = 2
epsilon = 1e-6

class Lick_Env_Cont(gym.Env):
    def __init__(self, action_dim, timesteps, thresh, dt):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.thresh = thresh
        self.max_timesteps = timesteps
        self.dt = dt
        self.cortical_state = 0
        self.switch = 0
        self.cue = 0
        self.cue_time = 1 / dt
        self.beta = 0.9

    def reset(self, episode) -> list:

        self.cue = 0
        self.cortical_state = 0
        # switch target delay time
        if episode % 1 == 0:
            if self.switch == 0:
                self.switch = 1
            else:
                self.switch = 0

        state = [self.cortical_state, self.switch, self.cue]
        return state
    
    def _get_reward(self, t: int, action: int, activity: int) -> int:

        if self.switch == 1:
            delay_time = 2 / self.dt
        else:
            delay_time = 4 / self.dt

        reward = 0
        if action == 1 and t >= delay_time:
            reward = delay_time / t

        return reward
    
    def _get_done(self, t: int, action: int) -> bool:
        if t == self.max_timesteps or action == 1:
            done = True
        else:
            done = False
        return done
    
    def _get_next_state(self, t: int) -> torch.Tensor:

        if t == self.cue_time:
            self.cue = 1
        else:
            self.cue = 0

        state = [self.cortical_state, self.switch, self.cue]
        return state
    
    def _get_lick(self, action: torch.Tensor) -> torch.Tensor:
        self.cortical_state = self.beta * self.cortical_state + action

        if self.cortical_state >= self.thresh:
            lick = 1
        else:
            lick = 0

        return lick
    
    def step(self, t: int, action: torch.Tensor) -> (list, int, bool):
        action = action[0]
        next_t = t+1
        lick = self._get_lick(action)
        state = self._get_next_state(next_t)
        reward = self._get_reward(next_t, lick, action)
        done = self._get_done(next_t, lick)
        return state, reward, done
    
    




