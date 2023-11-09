import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sac_model import Actor, Critic
from sac_learn import OptimizerSpec, sac_learn
from utils.gym import get_env, get_wrapper_by_name
from lick_env import Lick_Env_Cont

alm_activity = scipy.io.loadmat("warped_activity_time1s.mat")
alm_activity_arr = alm_activity["warped_trial_activity_1s"]
seconds = 2.
ms = seconds * 1000.
alm_activity_arr = alm_activity_arr[:, int(alm_activity_arr.shape[1]/2):int(alm_activity_arr.shape[1]/2)+int(ms)]

BATCH_SIZE = 32
INP_DIM = 1
HID_DIM = 256
ACTION_DIM = 256
ALPHA = 0.20
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 100
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGER_UPDATE_FREQ = 10000
LEARNING_RATE = 0.003
ALPHA_OPT = 0.95
EPS = 0.01
DT = 0.01
TARGET_TIME = 2.0
# skipping by 10 because we are simulating 10 millisecond timesteps
TARGET_DYNAMICS = alm_activity_arr[:, 0:-1:10]
THRESH = 0.01
ALM_HID = 256

def main(env, seed):

    optimizer_spec = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA_OPT, eps=EPS),
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
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        target_update_freq=TARGER_UPDATE_FREQ,
    )

if __name__ == '__main__':
    # Get Atari games.
    seed = 0 # Use a seed of zero (you may want to randomize the seed!)
    env = Lick_Env_Cont(ACTION_DIM, DT, TARGET_TIME, TARGET_DYNAMICS, THRESH, ALM_HID)

    # Run training
    env = get_env(env, seed)
    main(env, seed)