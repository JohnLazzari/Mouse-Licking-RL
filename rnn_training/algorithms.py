import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name
from sac_model import Actor, Critic

def select_action(policy: Actor, state: list, hn: torch.Tensor, evaluate: bool):
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
    hn = hn.cuda()

    if evaluate == False: 
        action, _, _, _, hn = policy.sample(state, hn, sampling=True)
    else:
        _, _, action, _, hn = policy.sample(state, hn, sampling=True)

    return action.detach().cpu().tolist()[0], hn.detach()

def soft_update(target: Critic, source: Critic, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target: Critic, source: Critic):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def sac(actor,
        critic,
        critic_target,
        critic_optimizer,
        actor_optimizer,
        policy_memory,
        batch_size,
        hid_dim,
        gamma,
        automatic_entropy_tuning,
        log_alpha,
        target_entropy,
        alpha,
        alpha_optim,
        ):

    if automatic_entropy_tuning:
        alpha = log_alpha.exp()

    # Sample a batch from memory
    state_batch, action_batch, reward_batch, next_state_batch, mask_batch = policy_memory.sample(batch_size)

    len_seq = list(map(len, state_batch))
    state_batch = pad_sequence(state_batch, batch_first=True).cuda()
    action_batch = pad_sequence(action_batch, batch_first=True).cuda()
    reward_batch = pad_sequence(reward_batch, batch_first=True).unsqueeze(-1).cuda()
    next_state_batch = pad_sequence(next_state_batch, batch_first=True).cuda()
    mask_batch = pad_sequence(mask_batch, batch_first=True).unsqueeze(-1).cuda()

    h_train = torch.zeros(size=(1, batch_size, hid_dim))
    with torch.no_grad():
        next_state_action, next_state_log_pi, _, _, _ = actor.sample(next_state_batch, h_train, sampling=False, len_seq=len_seq)
        qf1_next_target, qf2_next_target = critic_target(next_state_batch, next_state_action, h_train, len_seq)
        min_qf_next_target = torch.minimum(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
        next_q_value = reward_batch + mask_batch * gamma * (min_qf_next_target)

    # mask the losses which correspond to padded values (just in case)
    loss_mask = [torch.ones(size=(length, 1)) for length in len_seq]
    loss_mask = pad_sequence(loss_mask, batch_first=True).cuda()

    qf1, qf2 = critic(state_batch, action_batch, h_train, len_seq)  # Two Q-functions to mitigate positive bias in the policy improvement step

    qf1 = qf1 * loss_mask
    qf2 = qf2 * loss_mask
    next_q_value = next_q_value * loss_mask

    qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
    qf_loss = qf1_loss + qf2_loss

    critic_optimizer.zero_grad()
    qf_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 1)
    critic_optimizer.step()

    pi_action_bat, log_prob_bat, _, _, _ = actor.sample(state_batch, h_train, sampling= False, len_seq=len_seq)
    qf1_pi, qf2_pi = critic(state_batch, pi_action_bat, h_train, len_seq)

    qf1_pi = qf1_pi * loss_mask
    qf2_pi = qf2_pi * loss_mask
    log_prob_bat = log_prob_bat * loss_mask

    min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

    policy_loss = ((alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

    actor_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
    actor_optimizer.step()

    if automatic_entropy_tuning:
        alpha_loss = -(log_alpha * (log_prob_bat + target_entropy).detach()).mean()

        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

    soft_update(critic_target, critic, .005)

    return policy_loss.item(), qf_loss.item()


def REINFORCE(episode, alm, gamma, log_probs, alm_values):

    alm.alm_values_optim.zero_grad()
    alm.alm_optim.zero_grad()

    update_alm_values = []
    update_alm = []
    
    R = 0

    returns = deque()
    for tuple in episode[::-1]:
        R = tuple[2] + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)

    for log_prob, R, value, t in zip(log_probs, returns, alm_values, range(len(episode))):
        delta = R - value
        update_alm_values.append(-delta.detach()*value)
        update_alm.append(-delta.detach()*log_prob)
    
    update_alm_values = torch.sum(torch.stack(update_alm_values))
    update_alm = torch.sum(torch.stack(update_alm))

    update_alm_values.backward() 
    update_alm.backward()
    
    alm.alm_values_optim.step()
    alm.alm_optim.step()

def One_Step_AC(tuple, actor, critic, actor_optim, critic_optim, gamma, I, z_critic, z_actor):

    lambda_critic = .9
    lambda_actor = .9

    state = torch.FloatTensor(tuple[0], device="cuda").unsqueeze(0).unsqueeze(0)
    action = torch.FloatTensor(tuple[1], device="cuda").unsqueeze(0).unsqueeze(0)
    reward = torch.FloatTensor(tuple[2], device="cuda").unsqueeze(0)
    next_state = torch.FloatTensor(tuple[3], device="cuda").unsqueeze(0).unsqueeze(0)
    mask = torch.FloatTensor(tuple[4], device="cuda").unsqueeze(0)
    h_prev = tuple[0][5].cuda()
    h_next = tuple[0][6].cuda()

    delta = reward[-1] + gamma * mask[-1] * critic(next_state, h_prev) - critic(state, h_prev)

    # Critic Update
    critic_optim.zero_grad()
    z_critic_func = {}
    for param in z_critic:
        z_critic_func[param] = (gamma * lambda_critic * z_critic[param]).detach()
    critic_forward = critic(state, h_prev)
    critic_forward.backward()
    # update z_critic and gradients
    for name, param in critic.named_parameters():
        z_critic[name] = (z_critic_func[name] + param.grad).detach()
        param.grad = -delta.detach().squeeze() * (z_critic_func[name] + param.grad)

    # Actor Update
    actor_optim.zero_grad()
    z_actor_func = {}
    for param in z_actor:
        z_actor_func[param] = (gamma * lambda_actor * z_actor[param]).detach()
    _, log_prob, _, _, _ = actor(state, h_prev)
    cur_log_prob = log_prob[int(action[-1].item())]
    cur_log_prob.backward()
    for name, param in actor.named_parameters():
        z_actor[name] = (z_actor_func[name] + I * param.grad).detach()
        param.grad = -delta.detach().squeeze() * (z_actor_func[name] + I * param.grad)

    I = gamma * I

    actor_optim.step()
    critic_optim.step()

    return I, z_critic, z_actor