import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from ac_model import Actor, Critic
from learn import OptimizerSpec, sac_learn
from utils.gym import get_env, get_wrapper_by_name
from lick_env import Lick_Env
import torch

BATCH_SIZE = 6
HID_DIM = 256
ACTION_DIM = 2
THALAMIC_INP_DIM = 1
ALPHA = 0.20
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 15_000
LEARNING_STARTS = 1_000
SAVE_ITER = 100_000
LEARNING_FREQ = 1
LEARNING_RATE = 0.001
ALPHA_OPT = 0.95
EPS = 0.01
THRESH = 0.5
ENTROPY_TUNING = True
WEIGHT_DECAY = .001
DT = 0.1
TARGET_TIME = 14
THALAMOCORTICAL_DIM = 8

INP_DIM = THALAMIC_INP_DIM + ACTION_DIM + THALAMOCORTICAL_DIM

def main(env, seed):

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, eps=EPS, weight_decay=WEIGHT_DECAY),
    )

    sac_learn(
        env=env,
        seed=seed,
        inp_dim=INP_DIM,
        hid_dim=HID_DIM,
        action_dim=ACTION_DIM,
        actor=Actor,
        critic=Critic,
        optimizer_spec=optimizer_spec,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        alpha=ALPHA,
        gamma=GAMMA,
        automatic_entropy_tuning=ENTROPY_TUNING,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        save_iter=SAVE_ITER
    )

if __name__ == '__main__':

    seed = np.random.randint(0, high=123456) # Use a seed of zero (you may want to randomize the seed!)
    torch.manual_seed(seed)
    env = Lick_Env(seed, DT, TARGET_TIME, THALAMIC_INP_DIM, THALAMOCORTICAL_DIM)

    # Run training
    env = get_env(env, seed)
    main(env, seed)