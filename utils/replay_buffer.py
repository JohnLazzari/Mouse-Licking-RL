import numpy as np
import random
import torch
from itertools import chain

class ReplayBuffer(object):
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

        state_batch = [[list(element)[0] for element in sample]for sample in batch]
        state_batch = list(map(torch.FloatTensor, state_batch))

        action_batch = [[list(element)[1] for element in sample]for sample in batch]
        action_batch = list(map(torch.FloatTensor, action_batch))

        reward_batch = [[list(element)[2] for element in sample]for sample in batch]
        reward_batch = list(map(torch.FloatTensor, reward_batch))

        next_state_batch = [[list(element)[3] for element in sample]for sample in batch]
        next_state_batch = list(map(torch.FloatTensor, next_state_batch))

        done_batch = [[list(element)[4] for element in sample]for sample in batch]
        done_batch = list(map(torch.FloatTensor, done_batch))

        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
