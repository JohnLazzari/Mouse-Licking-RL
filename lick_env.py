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
        #nn.init.uniform_(self._alm.weight_ih_l0, -np.sqrt(6 / (alm_hid+action_dim)), np.sqrt(6 / (alm_hid+action_dim)))
        #nn.init.uniform_(self._alm.weight_ih_l0, -np.sqrt(6 / (alm_hid+action_dim)), np.sqrt(6 / (alm_hid+action_dim)))
        self._alm_out = nn.Linear(alm_hid, 2)
        #nn.init.uniform_(self._alm_out.weight, 0, np.sqrt(6 / (alm_hid+2)))
    
    def _get_reward(self, t, activity):
        mse = torch.abs(activity-torch.tensor(self._target_dynamics[t,:]))
        range = torch.any(mse > self._thresh).item()
        if range:
            reward = -1
        else:
            reward = 5*torch.sum(1 / (1000**mse+1e-6)).item()
        return reward, mse
    
    def _get_done(self, t, error):
        range = torch.any(error > self._thresh).item()
        if range or t == self.max_timesteps-1:
            done = True
        else:
            done = False
        return done
    
    def _get_next_state(self):
        self.time += self._dt
        state = torch.cat((self._alm_hn.squeeze(), torch.tensor(self.time).unsqueeze(0)))
        return state
    
    def _get_activity_hid(self, action):
        with torch.no_grad():
            activity, self._alm_hn = self._alm(torch.tensor(action).unsqueeze(0), self._alm_hn)
            activity = F.hardtanh(self._alm_out(activity), min_val=-.1, max_val=.2)
        return activity
    
    def reset(self):
        self._alm_hn = torch.zeros(size=(1, self._alm_hid))
        self.time = 0.
        state = torch.cat((self._alm_hn.squeeze(), torch.tensor(self.time).unsqueeze(0)))
        return state.tolist()

    def step(self, t, action):
        activity = self._get_activity_hid(action)
        state = self._get_next_state()
        reward, error = self._get_reward(t, activity)
        done = self._get_done(t, error)
        return state.tolist(), reward, done
    
    




