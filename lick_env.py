import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sac_model import weights_init_

class Lick_Env_Discrete(gym.Env):
    def __init__(self, seed, dt, target_time):
        super(Lick_Env_Discrete, self).__init__()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.dt = dt
        self.target_time = target_time
        self.max_timesteps = int(target_time / dt)+1

    def reset(self):
        self.state = 0
        return self.state
    
    def _get_reward_done(self, t, a_t):
        if t == self.max_timesteps-1:
            if a_t == 1:
                return 1, True
            elif a_t == 0:
                return -1, True
        else:
            if a_t == 1:
                return -1, True
            elif a_t == 0:
                return 0, False
    
    def _get_next_state(self):
        self.state += self.dt

    def step(self, t, action):
        reward, done = self._get_reward_done(t, action)
        self._get_next_state()
        return self.state, reward, done, None
    

class Lick_Env_Cont(gym.Env):
    def __init__(self, action_dim, target_dynamics, thresh, alm_hid):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # might change this to length of targ_dynamics but good to know the timescale
        self.thresh = thresh
        self._target_dynamics = torch.tensor(target_dynamics)
        self.max_timesteps = self._target_dynamics.shape[0]
        self._alm_hid = alm_hid
        self._alm = nn.RNN(action_dim, alm_hid, batch_first=True, nonlinearity='relu')
        self._alm_out = nn.Linear(alm_hid, 3)
    
    def _get_reward(self, t: int, activity: torch.Tensor) -> (int, torch.Tensor):
        mse = torch.abs(activity-self._target_dynamics[t,:])
        range = torch.any(mse > self.thresh).item()
        if range:
            reward = -10
        else:
            reward = 5*torch.sum(1 / (1000**mse+1e-6)).item()
        return reward, mse
    
    def _get_done(self, t: int, error: torch.Tensor) -> bool:
        range = torch.any(error > self.thresh).item()
        if range or t == self.max_timesteps-1:
            done = True
        else:
            done = False
        return done
    
    def _get_next_state(self, activity: torch.Tensor, t: int) -> torch.Tensor:
        state = torch.cat((self._alm_hn.squeeze(), activity.squeeze(), self._target_dynamics[t,:]))
        return state
    
    def _get_activity_hid(self, action: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            action = torch.tensor(action).unsqueeze(0)
            activity, self._alm_hn = self._alm(action, self._alm_hn)
            activity = F.hardtanh(self._alm_out(activity), min_val=0, max_val=1)
        return activity
    
    def reset(self) -> list:
        self._alm_hn = torch.zeros(size=(1, self._alm_hid))
        activity = torch.zeros(size=(3,))
        state = torch.cat((self._alm_hn.squeeze(), activity, self._target_dynamics[0,:]))
        return state.tolist()

    def step(self, t: int, action: torch.Tensor) -> (list, int, bool):
        if t < self.max_timesteps-1:
            next_t = t+1
        else:
            next_t = t
        activity = self._get_activity_hid(action)
        state = self._get_next_state(activity, next_t)
        reward, error = self._get_reward(t, activity)
        done = self._get_done(t, error)
        return state.tolist(), reward, done
    
    




