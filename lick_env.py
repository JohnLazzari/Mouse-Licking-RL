import gym
import numpy as np

class Lick_Env(gym.Env):
    def __init__(self, seed, dt, target_time):
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




