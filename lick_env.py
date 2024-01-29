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
        self.switch_const = 0
        self.cortical_state = 0
        self.switch = 0
        self.cue = 0
        self.cue_time = 1 / dt
        self.beta = 0.85

    def reset(self, episode: int) -> list:

        self.cue = 0
        self.cortical_state = 0
        # switch target delay time
        if episode % 1 == 0:
            if self.switch == 0:
                self.switch = 1
                self.switch_const = 0.2
            else:
                self.switch = 0
                self.switch_const = 0.3

        state = [self.cortical_state, self.switch_const, self.cue]
        return state
    
    def _get_reward(self, t: int, action: int) -> int:

        if self.switch == 1:
            delay_time = 2 / self.dt
        else:
            delay_time = 3 / self.dt

        reward = 0
        if self.cue == 1:
            if action == 1 and t >= delay_time:
                reward += 5 * (delay_time / t)
        elif self.cue == 0:
            if self.cortical_state < 0.1 and t >= delay_time:
                reward += 1

        return reward
    
    def _get_done(self, t: int, action: int) -> bool:

        if self.switch == 1:
            delay_time = 2 / self.dt
        else:
            delay_time = 3 / self.dt

        done = False
        if t == self.max_timesteps:
            done = True
        if t > delay_time and self.cue == 0 and self.cortical_state < 0.1:
            done = True
        if action == 1 and t < delay_time:
            done = True
        return done
    
    def _get_next_state(self, t: int, lick: int) -> torch.Tensor:

        if t == self.cue_time:
            self.cue = 1
        if lick == 1:
            self.cue = 0

        state = [self.cortical_state, self.switch_const, self.cue]
        return state
    
    def _get_lick(self, action: torch.Tensor) -> torch.Tensor:
        self.cortical_state = max(0, self.beta * self.cortical_state + action * 0.2)

        if self.cortical_state >= self.thresh:
            lick = 1
        else:
            lick = 0

        return lick
    
    def step(self, t: int, action: torch.Tensor) -> (list, int, bool):
        action = action[0]
        next_t = t+1
        lick = self._get_lick(action)
        reward = self._get_reward(next_t, lick)
        state = self._get_next_state(next_t, lick)
        done = self._get_done(next_t, lick)
        return state, reward, done
    
    




