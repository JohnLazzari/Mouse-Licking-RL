"""
    This file is copied/apdated from https://github.com/berkeleydeeprlcourse/homework/tree/master/hw3
"""
import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Variable(autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

"""
    OptimizerSpec containing following attributes
        constructor: The optimizer constructor ex: RMSprop
        kwargs: {Dict} arguments for constructing optimizer
"""
OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

Statistic = {
    "mean_episode_rewards": [],
    "best_mean_episode_rewards": []
}

def dqn_learing(
    env,
    seed,
    q_func,
    optimizer_spec,
    exploration,
    replay_buffer_size=1000000,
    batch_size=32,
    gamma=0.99,
    learning_starts=50000,
    learning_freq=4,
    frame_history_len=4,
    target_update_freq=10000
    ):

    """Run Deep Q-learning algorithm.

    You can specify your own convnet using q_func.

    All schedules are w.r.t. total number of steps taken in the environment.

    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            input_channel: int
                number of channel of input.
            num_actions: int
                number of actions
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    exploration: Schedule (defined in utils.schedule)
        schedule for probability of chosing random action.
    stopping_criterion: (env) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    input_arg = env.observation_space.shape[0]
    num_actions = env.action_space.n

    # Construct an epilson greedy policy with given exploration schedule
    def select_epilson_greedy_action(model, obs, t, hn):
        sample = random.random()
        eps_threshold = exploration.value(t)
        if sample > eps_threshold:
            with torch.no_grad():
                obs = torch.tensor(obs).type(dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                # Use volatile = True if variable is only used in inference mode, i.e. don’t save the history
                out, hn = model(obs.cuda(), hn)
                return out.squeeze().max(-1)[1].item(), hn
        else:
            with torch.no_grad():
                obs = torch.tensor(obs).type(dtype).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                _, hn = model(obs.cuda(), hn)
                return random.randrange(num_actions), hn

    # Initialize target q function and q function
    Q = q_func(input_arg, num_actions).type(dtype)
    target_Q = q_func(input_arg, num_actions).type(dtype)

    # Construct Q network optimizer function
    optimizer = optimizer_spec.constructor(Q.parameters(), **optimizer_spec.kwargs)

    # Construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, seed)

    ###############
    # RUN ENV     #
    ###############
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    avg_reward = []
    ep_trajectory = []
    last_obs = env.reset()
    hn = torch.zeros(size=(1, 1, 256)).cuda()
    LOG_EVERY_N_STEPS = 10000

    for t in count():

        # Choose random action if not yet start learning
        if t > learning_starts:
            action, hn = select_epilson_greedy_action(Q, last_obs, t, hn)
        else:
            action = random.randrange(num_actions)
        # Advance one step
        obs, reward, done, _ = env.step(t%env.max_timesteps, action)
        avg_reward.append(reward)
        print(np.mean(np.array(avg_reward)))
        # Store other info in replay memory
        ep_trajectory.append([last_obs, action, reward, obs, done])
        # Resets the environment when reaching an episode boundary.
        if done:
            obs = env.reset()
            replay_buffer.push(ep_trajectory)
            ep_trajectory = []
            hn = torch.zeros(size=(1, 1, 256)).cuda()
        last_obs = obs

        ### Perform experience replay and train the network.
        # Note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                len(replay_buffer.buffer) > batch_size):
            # Use the replay buffer to sample a batch of transitions
            # Note: done_mask[i] is 1 if the next state corresponds to the end of an episode,
            # in which case there is no Q-value at the next state; at the end of an
            # episode, only the current state reward contributes to the target
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_buffer.sample(batch_size)

            obs_batch = pad_sequence(obs_batch, batch_first=True).unsqueeze(-1)
            act_batch = pad_sequence(act_batch, batch_first=True).unsqueeze(-1).type(torch.int64)
            rew_batch = pad_sequence(rew_batch, batch_first=True).unsqueeze(-1)
            next_obs_batch = pad_sequence(next_obs_batch, batch_first=True).unsqueeze(-1)
            not_done_mask = 1 - pad_sequence(done_mask, batch_first=True).unsqueeze(-1)

            if USE_CUDA:
                next_obs_batch = next_obs_batch.cuda()
                obs_batch = obs_batch.cuda()
                act_batch = act_batch.cuda()
                rew_batch = rew_batch.cuda()
                not_done_mask = not_done_mask.cuda()

            # Compute current Q value, q_func takes only state and output value for every state-action pair
            # We choose Q based on action taken.
            h_train = torch.zeros(size=(1, batch_size, 256)).cuda()
            current_Q_values, _ = Q(obs_batch, h_train)
            current_Q_values = current_Q_values.gather(2, act_batch)
            # Compute next Q value based on which action gives max Q values
            # Detach variable from the current graph since we don't want gradients for next Q to propagated
            h_train = torch.zeros(size=(1, batch_size, 256)).cuda()
            next_max_q, _ = target_Q(next_obs_batch, h_train)
            next_max_q = next_max_q.detach().max(2)[0].unsqueeze(-1)
            next_Q_values = not_done_mask * next_max_q
            # Compute the target of the current Q values
            target_Q_values = rew_batch + (gamma * next_Q_values)
            # Compute Bellman error
            bellman_error = target_Q_values - current_Q_values
            # clip the bellman error between [-1 , 1]
            clipped_bellman_error = bellman_error.clamp(-1, 1)
            # Note: clipped_bellman_delta * -1 will be right gradient
            d_error = clipped_bellman_error * -1.0
            # Clear previous gradients before backward pass
            optimizer.zero_grad()
            # run backward pass
            current_Q_values.backward(d_error.data)

            # Perfom the update
            optimizer.step()
            num_param_updates += 1

            # Periodically update the target network by Q network to target Q network
            if num_param_updates % target_update_freq == 0:
                target_Q.load_state_dict(Q.state_dict())
