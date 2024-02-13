import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import scipy.io as sio

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

class Lick_Env_Cont(gym.Env):
    def __init__(self, action_dim, timesteps, thresh, dt, beta, bg_scale, alm_data_path):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.thresh = thresh
        self.max_timesteps = timesteps
        self.dt = dt
        self.switch_const = 0
        self.cortical_state = 0
        self.switch = 1
        self.cue = 0
        self.cue_time = 1 / dt
        self.beta = beta
        self.bg_scale = bg_scale
        self.alm_data_path = alm_data_path
        self.time_elapsed_from_lick = 0

        # Load data
        self.alm_activity = sio.loadmat(alm_data_path)['average_total_fr_units_1s']
        self.alm_activity = np.squeeze(NormalizeData(self.alm_activity))

    # TODO potentially increase the timesteps with frameskipping as well, build visualization tool with pygame, debug new environment
    def reset(self, episode: int) -> list:

        self.cue = 0
        self.cortical_state = 0
        self.time_elapsed_from_lick = 0
        # switch target delay time
        '''
        if episode % 1 == 0:
            if self.switch == 0:
                self.switch = 1
                self.switch_const = 0.2
            else:
                self.switch = 0
                self.switch_const = 0.3
        '''
        self.switch = 1
        self.switch_const = 0.2

        state = [self.cortical_state, self.switch_const, self.cue]
        return state
    
    def _get_reward(self, t: int, action: int, activity: int, y_depression: torch.Tensor) -> int:

        if self.switch == 1:
            delay_time = 2 / self.dt
        else:
            delay_time = 3 / self.dt

        reward = 0
        #reward -= torch.linalg.norm(y_depression).item()
        if self.cue == 1:

            # Follow the ramping activity while the cue has sounded and the mouse hasnt licked yet
            # Goal is to further incentivize accurate ramping ramping activity
            reward -= 0.01 * abs(activity - self.alm_activity[t-1])
            # Provide a high reward once ramping is successfully completed after the delay time 
            # Reward is scaled by how late the mouse licks
            if action == 1 and t >= delay_time:
                reward += 5 * (delay_time / t)
            if action == 1 and t < delay_time:
                reward -= 5
            if action != 1 and t == self.max_timesteps:
                reward -= 5

        elif self.cue == 0:

            '''
            # If cue is zero and t is past the delay time, that means the mouse licked already and now the task is to decrease activity
            # Incentivizing mouse to follow true decay activity
            if t > delay_time:
                # The position in the alm activity should start from peak in this case
                reward -= 0.01 * abs(activity - self.alm_activity[(int(delay_time)-1) + self.time_elapsed_from_lick])
                self.time_elapsed_from_lick += 1

            # If the alm activity decays to around 0.5, end the episode and give high reward
            if self.cortical_state <= self.alm_activity[-1] and t > delay_time:
                reward += 5
            '''

            # If cue is zero and t is less than cue time, this is pre-cue activity, thus follow true trajectory in order to reduce alm activity before cue
            if t <= self.cue_time:
                reward -= 0.01 * abs(activity - self.alm_activity[t-1])
                if action == 1:
                    reward -= 5

        return reward
    
    def _get_done(self, t: int, action: int) -> bool:

        if self.switch == 1:
            delay_time = 2 / self.dt
        else:
            delay_time = 3 / self.dt

        done = False
        if t == self.max_timesteps:
            done = True
        #if t > delay_time and self.cue == 0 and self.cortical_state <= self.alm_activity[-1]:
        #    done = True
        if action == 1:
            done = True
        return done
    
    def _get_next_state(self, t: int, lick: int) -> torch.Tensor:

        if t == self.cue_time:
            self.cue = 1
        #if lick == 1:
        #    self.cue = 0

        state = [self.cortical_state, self.switch_const, self.cue]
        return state
    
    def _get_lick(self, action: torch.Tensor) -> torch.Tensor:
        self.cortical_state = max(0, self.beta * self.cortical_state + action * self.bg_scale)

        if self.cortical_state >= self.thresh:
            lick = 1
            self.cortical_state = 1
        else:
            lick = 0

        return lick
    
    def step(self, t: int, action: torch.Tensor, hn: torch.Tensor, y_depression) -> (list, int, bool):
        action = action[0]
        next_t = t+1
        lick = self._get_lick(action)
        reward = self._get_reward(next_t, lick, action, y_depression)
        done = self._get_done(next_t, lick)
        state = self._get_next_state(next_t, lick)
        return state, reward, done
    
    
class Trajectory_Env(gym.Env):
    def __init__(self, action_dim, timesteps, dt, beta, bg_scale, alm_data_path):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.max_timesteps = timesteps
        self.dt = dt
        self.switch_const = 0
        self.cortical_state = 0
        self.cue = 0
        self.cue_time = 1 / dt
        self.beta = beta
        self.bg_scale = bg_scale
        self.alm_data_path = alm_data_path
        self.thresh = 0.05

        # Load data
        self.alm_activity = sio.loadmat(alm_data_path)['average_total_fr_units_1s']
        self.alm_activity = np.squeeze(NormalizeData(self.alm_activity))

    def reset(self, episode: int) -> list:

        self.cue = 0
        self.cortical_state = 0
        self.switch_const = 0.2

        state = [self.cortical_state, self.switch_const, self.cue]
        return state
    
    def _get_reward(self, t: int, activity: int) -> int:

        dist = abs(activity - self.alm_activity[t-1])
        reward = 5 * (1 / 1000**(dist))

        return reward
    
    def _get_done(self, t: int, activity: int) -> bool:

        done = False
        dist = abs(activity - self.alm_activity[t-1])
        if dist > self.thresh:
            done = True
        if t == self.max_timesteps:
            done = True
        return done
    
    def _get_next_state(self, t: int) -> torch.Tensor:

        # hardcoded for now
        if t == self.cue_time:
            self.cue = 1
        if t == 200:
            self.cue = 0

        state = [self.cortical_state, self.switch_const, self.cue]
        return state
    
    def _get_activity(self, action: torch.Tensor) -> torch.Tensor:
        self.cortical_state = max(0, self.beta * self.cortical_state + action * self.bg_scale)
    
    def step(self, t: int, action: torch.Tensor, hn: torch.Tensor) -> (list, int, bool):
        action = action[0]
        next_t = t+1
        self._get_activity(action)
        reward = self._get_reward(next_t, action)
        state = self._get_next_state(next_t)
        done = self._get_done(next_t, action)
        return state, reward, done