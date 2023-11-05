import numpy as np
import random
import torch
from itertools import chain

class PolicyBuffer(object):
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = state
        self.position = (self.position + 1) % self.capacity

    def __len__(self):
        return len(self.buffer)
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        batch_list = list(chain(*batch))
        state, action, reward, next_state, done, h_current, c_current = map(np.stack, zip(*batch_list))

        policy_state_batch = [[list(element)[0] for element in sample]for sample in batch]
        policy_state_batch = list(map(torch.FloatTensor, policy_state_batch))

        return state, action, reward, next_state, done, h_current, c_current, policy_state_batch
