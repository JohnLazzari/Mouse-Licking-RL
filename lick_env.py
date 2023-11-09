import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, action_dim, dt, target_time, target_dynamics, thresh, alm_hid):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        # might change this to length of targ_dynamics but good to know the timescale
        self.max_timesteps = int(target_time / dt)
        self._thresh = thresh
        self._dt = dt
        self._target_time = target_time
        self._target_dynamics = target_dynamics
        self._alm_hid = alm_hid
        self._alm = nn.GRU(action_dim, alm_hid, batch_first=True)
        self._alm_out = nn.Linear(alm_hid, 30)
    
    def _get_reward(self, t, action):
        with torch.no_grad():
            activity, self._alm_hn = self._alm(torch.tensor(action).unsqueeze(0), self._alm_hn)
            activity = F.relu(self._alm_out(activity))
        mse = torch.mean((activity-torch.tensor(self._target_dynamics[:,t]))**2).item()
        reward = .05 * (1 / mse)
        return reward, mse
    
    def _get_done(self, t, error):
        if error > self._thresh or t == self.max_timesteps-1:
            done = True
        else:
            done = False
        return done
    
    def _get_next_state(self):
        # TODO make the state more than just time, but also hidden state of alm GRU
        self.state += self._dt
    
    def reset(self):
        self._alm_hn = torch.zeros(size=(1, self._alm_hid))
        self.state = 0.
        return self.state

    def step(self, t, action):
        reward, error = self._get_reward(t, action)
        done = self._get_done(t, error)
        self._get_next_state()
        return self.state, reward, done, None
    
    




