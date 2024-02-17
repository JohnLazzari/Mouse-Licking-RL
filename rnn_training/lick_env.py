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
    return 0.9 * ( (data - min) / (max - min) ) + 0.1

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
    
    def _get_reward(self, t: int, action: int, activity: int, hn: torch.Tensor) -> int:

        if self.switch == 1:
            delay_time = 2 / self.dt
        else:
            delay_time = 3 / self.dt

        reward = 0
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
                reward -= 0.01 * abs(activity - self.alm_activity[t-1] + 1e-2)
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
    
    def step(self, t: int, action: torch.Tensor, hn: torch.Tensor) -> (list, int, bool):
        action = action[0]
        next_t = t+1
        lick = self._get_lick(action)
        reward = self._get_reward(next_t, lick, action, hn)
        done = self._get_done(next_t, lick)
        state = self._get_next_state(next_t, lick)
        return state, reward, done
    

class Kinematics_Env(gym.Env):
    def __init__(self, action_dim, dt, kinematics_folder):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_dim = action_dim
        self.dt = dt
        self.cue = 0
        self.cue_time = 1 / dt
        self.kinematics_folder = kinematics_folder
        self.thresh = 0.1
        self.fixed_steps = 1
        self.max_timesteps = None
        self.cur_cond = None
        self.cortical_state = np.ones(shape=(action_dim,)) * 0.1
        self.kinematics_jaw_x = {}
        self.kinematics_jaw_y = {}
        self.kinematics_tongue_x = {}
        self.kinematics_tongue_y = {}
        self.Taxis = {}

        # Load data
        for cond in range(3):

            self.kinematics_jaw_y[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}y_jaw.mat')['condy_jaw_mean']
            self.kinematics_jaw_x[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}x_jaw.mat')['condx_jaw_mean']
            # y position is lower than x position, using these min and max values such that the scaling between x and y is accurate
            min_jaw_y, max_jaw_y = np.min(self.kinematics_jaw_y[cond]), np.max(self.kinematics_jaw_y[cond])
            y_diff = max_jaw_y - min_jaw_y
            # we want to have them be between 0 and 1 but at a reasonable scale
            min_jaw_x, max_jaw_x = np.min(self.kinematics_jaw_x[cond]), np.min(self.kinematics_jaw_x[cond]) + y_diff

            self.kinematics_jaw_y[cond] = NormalizeData(np.squeeze(self.kinematics_jaw_y[cond]), min_jaw_y, max_jaw_y)
            self.kinematics_jaw_x[cond] = NormalizeData(np.squeeze(self.kinematics_jaw_x[cond]), min_jaw_x, max_jaw_x)

            plt.plot(self.kinematics_jaw_y[cond])
            plt.plot(self.kinematics_jaw_x[cond])
            plt.show()

            self.kinematics_tongue_y[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}y_tongue.mat')['condy_tongue_mean']
            self.kinematics_tongue_x[cond] = sio.loadmat(f'{kinematics_folder}/cond{cond+1}x_tongue.mat')['condx_tongue_mean']
            min_tongue_y, max_tongue_y = np.min(self.kinematics_tongue_y[cond]), np.max(self.kinematics_tongue_y[cond])
            y_diff = max_tongue_y - min_tongue_y
            min_tongue_x, max_tongue_x = np.min(self.kinematics_tongue_x[cond]), np.min(self.kinematics_tongue_x[cond]) + y_diff

            self.kinematics_tongue_y[cond] = NormalizeData(np.squeeze(self.kinematics_tongue_y[cond]), min_tongue_y, max_tongue_y)
            self.kinematics_tongue_x[cond] = NormalizeData(np.squeeze(self.kinematics_tongue_x[cond]), min_tongue_x, max_tongue_x)

            plt.plot(self.kinematics_tongue_y[cond])
            plt.plot(self.kinematics_tongue_x[cond])
            plt.show()

            self.Taxis[cond] = sio.loadmat(f'{kinematics_folder}/Taxis_cond{cond+1}.mat')['Taxis_cur'].squeeze()

    def reset(self, episode: int) -> list:

        self.cur_cond = episode % 3
        assert self.kinematics_jaw_x[self.cur_cond].shape == self.kinematics_jaw_y[self.cur_cond].shape
        assert self.kinematics_tongue_x[self.cur_cond].shape == self.kinematics_tongue_y[self.cur_cond].shape
        self.max_timesteps = self.kinematics_jaw_x[self.cur_cond].shape[0]
        self.speed_const = (self.cur_cond + 1) / 3
        self.cue = 0
        self.thresh = 0.1
        self.cortical_state = np.ones(shape=(self.action_dim,)) * 0.1

        # [pred_x_pos, pred_y_pos, true_x_pos, true_y_pos, speed_const, cue]
        state = [0., 
                0., 
                0.,
                0.,
                self.kinematics_jaw_x[self.cur_cond][0], 
                self.kinematics_jaw_y[self.cur_cond][0], 
                self.kinematics_tongue_x[self.cur_cond][0], 
                self.kinematics_tongue_y[self.cur_cond][0], 
                self.speed_const, 
                self.cue]

        return state
    
    def _get_reward(self, t: int) -> int:

        dist_x_jaw = abs(self.cortical_state[0] - self.kinematics_jaw_x[self.cur_cond][t])
        dist_y_jaw = abs(self.cortical_state[1] - self.kinematics_jaw_y[self.cur_cond][t])

        dist_x_tongue = abs(self.cortical_state[2] - self.kinematics_tongue_x[self.cur_cond][t])
        dist_y_tongue = abs(self.cortical_state[3] - self.kinematics_tongue_y[self.cur_cond][t])

        if dist_x_jaw > self.thresh or dist_y_jaw > self.thresh or dist_x_tongue > self.thresh or dist_y_tongue > self.thresh:
            reward = 5 * (-dist_x_jaw - dist_y_jaw - dist_x_tongue - dist_y_tongue)
            return reward

        reward_x_jaw = (1 / 1000**(dist_x_jaw))
        reward_y_jaw = (1 / 1000**(dist_y_jaw))

        reward_x_tongue = (1 / 1000**(dist_x_tongue))
        reward_y_tongue = (1 / 1000**(dist_y_tongue))

        reward = reward_x_jaw + reward_y_jaw + reward_x_tongue + reward_y_tongue

        # add reward based on cue
        if -0.033 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 0.033:
            reward += 5

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
    
    def _get_done(self, t: int) -> bool:

        done = False

        dist_x_jaw = abs(self.cortical_state[0] - self.kinematics_jaw_x[self.cur_cond][t])
        dist_y_jaw = abs(self.cortical_state[1] - self.kinematics_jaw_y[self.cur_cond][t])

        dist_x_tongue = abs(self.cortical_state[2] - self.kinematics_tongue_x[self.cur_cond][t])
        dist_y_tongue = abs(self.cortical_state[3] - self.kinematics_tongue_y[self.cur_cond][t])

        if dist_x_jaw > self.thresh or dist_y_jaw > self.thresh or dist_x_tongue > self.thresh or dist_y_tongue > self.thresh:
            done = True
        if t == self.max_timesteps-1:
            done = True
        return done
    
    def _get_next_state(self, t: int) -> torch.Tensor:

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
                self.cortical_state[2],
                self.cortical_state[3],
                self.kinematics_jaw_x[self.cur_cond][t], 
                self.kinematics_jaw_y[self.cur_cond][t], 
                self.kinematics_tongue_x[self.cur_cond][t], 
                self.kinematics_tongue_y[self.cur_cond][t], 
                self.speed_const, 
                self.cue]

        return state
    
    def _get_pred_kinematics(self, action):
        action = np.array(action)
        self.cortical_state = np.maximum(0, self.cortical_state + action)

    def step(self, t: int, action: torch.Tensor, hn: torch.Tensor, episode_num: int) -> (list, int, bool):

        self._get_pred_kinematics(action)
        reward = self._get_reward(t)
        done = self._get_done(t)
        state = self._get_next_state(t)
        return state, reward, done

    
class Kinematics_Jaw_Env(gym.Env):
    def __init__(self, action_dim, dt, kinematics_folder):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.action_dim = action_dim
        self.dt = dt
        self.cue = 0
        self.cue_time = 1 / dt
        self.kinematics_folder = kinematics_folder
        self.thresh = 0.1
        self.fixed_steps = 1
        self.max_timesteps = None
        self.cur_cond = None
        self.cortical_state = np.ones(shape=(action_dim,))*0.1
        self.kinematics_jaw_x = {}
        self.kinematics_jaw_y = {}
        self.kinematics_tongue_x = {}
        self.kinematics_tongue_y = {}
        self.Taxis = {}

        # Load data
        for cond in range(3):

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

    def reset(self, episode: int) -> list:

        self.cur_cond = episode % 3
        assert self.kinematics_jaw_x[self.cur_cond].shape == self.kinematics_jaw_y[self.cur_cond].shape
        self.max_timesteps = self.kinematics_jaw_x[self.cur_cond].shape[0]
        self.speed_const = (self.cur_cond + 1) / 3
        self.cue = 0
        self.thresh = 0.1
        self.cortical_state = np.ones(shape=(self.action_dim,))*0.1

        # [pred_x_pos, pred_y_pos, true_x_pos, true_y_pos, speed_const, cue]
        state = [0., 
                0., 
                self.kinematics_jaw_x[self.cur_cond][0], 
                self.kinematics_jaw_y[self.cur_cond][0], 
                self.speed_const, 
                self.cue]

        return state
    
    def _get_reward(self, t: int) -> int:

        dist_x_jaw = abs(self.cortical_state[0] - self.kinematics_jaw_x[self.cur_cond][t])
        dist_y_jaw = abs(self.cortical_state[1] - self.kinematics_jaw_y[self.cur_cond][t])

        if dist_x_jaw > self.thresh or dist_y_jaw > self.thresh:
            reward = 5 * (-dist_x_jaw - dist_y_jaw)
            return reward

        reward_x_jaw = (1 / 1000**(dist_x_jaw))
        reward_y_jaw = (1 / 1000**(dist_y_jaw))

        reward = reward_x_jaw + reward_y_jaw

        # add reward based on cue
        if -0.033 < self.Taxis[self.cur_cond][t] and self.Taxis[self.cur_cond][t] < 0.033:
            reward += 5

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
    
    def _get_done(self, t: int) -> bool:

        done = False

        dist_x_jaw = abs(self.cortical_state[0] - self.kinematics_jaw_x[self.cur_cond][t])
        dist_y_jaw = abs(self.cortical_state[1] - self.kinematics_jaw_y[self.cur_cond][t])

        if dist_x_jaw > self.thresh or dist_y_jaw > self.thresh:
            done = True
        if t == self.max_timesteps-1:
            done = True
        return done
    
    def _get_next_state(self, t: int) -> torch.Tensor:

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
                self.kinematics_jaw_x[self.cur_cond][t], 
                self.kinematics_jaw_y[self.cur_cond][t], 
                self.speed_const, 
                self.cue]

        return state
    
    def _get_pred_kinematics(self, action):
        action = np.array(action)
        self.cortical_state = np.maximum(0, self.cortical_state + action)

    def step(self, t: int, action: torch.Tensor, hn: torch.Tensor, episode_num: int) -> (list, int, bool):

        self._get_pred_kinematics(action)
        reward = self._get_reward(t)
        done = self._get_done(t)
        state = self._get_next_state(t)
        return state, reward, done
    