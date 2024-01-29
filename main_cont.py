import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sac_model import Actor, Critic
from sac_learn import OptimizerSpec, sac_learn
from utils.gym import get_env, get_wrapper_by_name
from lick_env import Lick_Env_Cont
import torch

BATCH_SIZE = 4
INP_DIM = 3
HID_DIM = 32
ACTION_DIM = 1
ALPHA = 0.20
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 50_000
LEARNING_STARTS = 100
SAVE_ITER = 5000
LEARNING_FREQ = 1
LEARNING_RATE = 0.0003
ALPHA_OPT = 0.95
EPS = 0.01
ENTROPY_TUNING = True
WEIGHT_DECAY = 0
DT = 0.1
TIMESTEPS = int(5 / DT)
THRESH = 1

def main(env, seed):

    optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY),
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
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    torch.manual_seed(seed)
    env = Lick_Env_Cont(ACTION_DIM, TIMESTEPS, THRESH, DT)

    # Run training
    env = get_env(env, seed)
    main(env, seed)