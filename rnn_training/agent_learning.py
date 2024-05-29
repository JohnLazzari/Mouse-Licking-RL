import sys
import pickle
import numpy as np
from collections import namedtuple
from itertools import count
import random
import gym.spaces

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_padded_sequence

from utils.replay_buffer import ReplayBuffer
from utils.gym import get_wrapper_by_name
from agent_models import Actor, Critic, Value
import scipy.io as sio
import matplotlib.pyplot as plt

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])

class Agent_Trainer():
    def __init__(self,
                env,
                seed,
                inp_dim,
                hid_dim,
                action_dim,
                optimizer_spec_actor,
                optimizer_spec_critic,
                replay_buffer_size,
                batch_size,
                alpha,
                gamma,
                automatic_entropy_tuning, 
                learning_starts,
                learning_freq,
                save_iter,
                log_steps,
                frame_skips,
                model_save_path,
                reward_save_path,
                steps_save_path,
                action_scale,
                action_bias,
                policy_type,
                update_iters,
                ):

        self.env = env
        self.seed = seed
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.action_dim = action_dim
        self.optimizer_spec_actor = optimizer_spec_actor
        self.optimizer_spec_critic = optimizer_spec_critic
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.gamma = gamma
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.save_iter = save_iter
        self.log_steps = log_steps
        self.frame_skips = frame_skips
        self.model_save_path = model_save_path
        self.reward_save_path = reward_save_path
        self.steps_save_path = steps_save_path
        self.action_scale = action_scale
        self.action_bias = action_bias
        self.policy_type = policy_type
        self.update_iters = update_iters

        self.striatum_data = sio.loadmat(f'data/firing_rates/striatum_fr_population_cond1.mat')['fr_population']
        self.striatum_data = self.Normalize_Data(np.squeeze(self.striatum_data), np.min(self.striatum_data), np.max(self.striatum_data))

    def Normalize_Data(self, data, min, max):
        '''
            Mainly used for neural data if model is constrained
        '''
        return (data - min) / (max - min)

    def select_action(self, policy, state, hn, evaluate):
        '''
            Selection of action from policy, consistent across training methods
        '''

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).cuda()
        hn = hn.cuda()

        if evaluate == False: 
            action, _, _, _, hn = policy.sample(state, hn, sampling=True)
        else:
            _, _, action, _, hn = policy.sample(state, hn, sampling=True)

        return action.detach().cpu().tolist()[0], hn.detach()

    def train(self):
        pass
    
    def update(self):
        pass


class On_Policy_Agent(Agent_Trainer):
    def __init__(self, 
                env,
                seed,
                inp_dim,
                hid_dim,
                action_dim,
                optimizer_spec_actor,
                optimizer_spec_critic,
                replay_buffer_size,
                batch_size,
                alpha,
                gamma,
                automatic_entropy_tuning, 
                learning_starts,
                learning_freq,
                save_iter,
                log_steps,
                frame_skips,
                model_save_path,
                reward_save_path,
                steps_save_path,
                action_scale,
                action_bias,
                policy_type,
                update_iters):

        super().__init__(env,
                        seed,
                        inp_dim,
                        hid_dim,
                        action_dim,
                        optimizer_spec_actor,
                        optimizer_spec_critic,
                        replay_buffer_size,
                        batch_size,
                        alpha,
                        gamma,
                        automatic_entropy_tuning, 
                        learning_starts,
                        learning_freq,
                        save_iter,
                        log_steps,
                        frame_skips,
                        model_save_path,
                        reward_save_path,
                        steps_save_path,
                        action_scale,
                        action_bias,
                        policy_type,
                        update_iters)
    
    def train(self, max_steps):

        '''
            Train the agent using one step actor critic
        '''

        actor_bg = Actor(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias).cuda()
        critic_bg = Value(self.inp_dim, self.hid_dim).cuda()

        actor_bg_optimizer = self.optimizer_spec_actor.constructor(actor_bg.parameters(), **self.optimizer_spec_actor.kwargs)
        critic_bg_optimizer = self.optimizer_spec_critic.constructor(critic_bg.parameters(), **self.optimizer_spec_critic.kwargs)

        z_actor = {}
        z_critic = {}
        I = 1
        for name, params in actor_bg.named_parameters():
            z_actor[name] = torch.zeros_like(params)
        for name, params in critic_bg.named_parameters():
            z_critic[name] = torch.zeros_like(params)

        Statistics = {
            "mean_episode_rewards": [],
            "mean_episode_steps": [],
            "best_mean_episode_rewards": []
        }

        episode_reward = 0
        best_mean_episode_reward = -float("inf")
        episode_steps = 0
        total_episodes = 0
        avg_reward = [0]
        avg_steps = [0]

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset(0)
        ep_trajectory = []

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hid_dim), device="cuda")

        ### STEPS PER EPISODE ###
        for t in range(max_steps):

            with torch.no_grad():
                action, h_current = self.select_action(actor_bg, state, h_prev, evaluate=False)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(self.frame_skips):
                next_state, reward, done = self.env.step(episode_steps, action, total_episodes)
                episode_steps += 1
                episode_reward += reward
                if done == True:
                    break

            mask = 1.0 if episode_steps == self.env.max_timesteps else float(not done)

            ep_trajectory.append((state, action, reward, next_state, mask))

            state = next_state
            h_prev = h_current

            ### EARLY TERMINATION OF EPISODE
            if done:

                total_episodes += 1

                # Add stats to lists
                avg_steps.append(episode_steps)
                avg_reward.append(episode_reward)

                # reset training conditions
                h_prev = torch.zeros(size=(1, 1, self.hid_dim), device="cuda")
                state = self.env.reset(total_episodes) 

                # resest lists
                ep_trajectory = []

                # reset eligibility trace (if using on-policy method)
                z_actor = {}
                z_critic = {}
                I = 1
                for name, params in actor_bg.named_parameters():
                    z_actor[name] = torch.zeros_like(params)
                for name, params in critic_bg.named_parameters():
                    z_critic[name] = torch.zeros_like(params)

                ### 4. Log progress and keep track of statistics
                if len(avg_reward) > 0:
                    mean_episode_reward = np.mean(np.array(avg_reward)[-1000:])
                if len(avg_steps) > 0:
                    mean_episode_steps = np.mean(np.array(avg_steps)[-1000:])
                if len(avg_reward) > 10:
                    if mean_episode_reward > best_mean_episode_reward:
                        torch.save({
                            'iteration': t,
                            'agent_state_dict': actor_bg.state_dict(),
                            'critic_state_dict': critic_bg.state_dict(),
                            'agent_optimizer_state_dict': actor_bg_optimizer.state_dict(),
                            'critic_optimizer_state_dict': critic_bg_optimizer.state_dict(),
                        }, self.model_save_path + '.pth')

                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                Statistics["mean_episode_rewards"].append(mean_episode_reward)
                Statistics["mean_episode_steps"].append(mean_episode_steps)
                Statistics["best_mean_episode_rewards"].append(best_mean_episode_reward)

                print("Episode %d" % (total_episodes,))
                print("reward: %f" % episode_reward)
                print("steps: %f" % episode_steps)
                print("best mean reward: %f" % best_mean_episode_reward)
                sys.stdout.flush()

                if total_episodes % self.log_steps == 0:
                    # Dump statistics to pickle
                    np.save(f'{self.reward_save_path}.npy', Statistics["mean_episode_rewards"])
                    np.save(f'{self.steps_save_path}.npy', Statistics["mean_episode_steps"])
                    print("Saved to %s" % 'training_reports')
                
                # reset tracking variables
                episode_steps = 0
                episode_reward = 0

            if done == False:
                I, z_critic, z_actor = self.update(ep_trajectory,
                                                    actor_bg,
                                                    critic_bg,
                                                    actor_bg_optimizer,
                                                    critic_bg_optimizer,
                                                    self.gamma,
                                                    I,
                                                    z_critic,
                                                    z_actor,
                                                    self.hid_dim)
    
    def update(self,
                tuple, 
                actor, 
                value, 
                actor_optim, 
                critic_optim, 
                gamma, 
                I, 
                z_critic, 
                z_actor,
                hid_dim):

        lambda_critic = .5
        lambda_actor = .5

        state = torch.tensor([step[0] for step in tuple], device="cuda").unsqueeze(0)
        action = torch.tensor([step[1] for step in tuple], device="cuda").unsqueeze(0)
        reward = torch.tensor([step[2] for step in tuple], device="cuda").unsqueeze(1)
        next_state = torch.tensor([step[3] for step in tuple], device="cuda").unsqueeze(0)
        mask = torch.tensor([step[4] for step in tuple], device="cuda").unsqueeze(1)

        h_update = torch.zeros(size=(1, 1, hid_dim), device="cuda")

        delta = reward + gamma * mask * value(next_state, h_update) - value(state, h_update)
        # TODO try either summing all deltas or only using last one
        delta = delta.squeeze(0)[-1]

        # Critic Update
        critic_optim.zero_grad()
        z_critic_func = {}
        for param in z_critic:
            z_critic_func[param] = (gamma * lambda_critic * z_critic[param]).detach()
        critic_forward = value(state, h_update)
        critic_forward = torch.sum(critic_forward.squeeze())
        critic_forward.backward()
        # update z_critic and gradients
        for name, param in value.named_parameters():
            z_critic[name] = (z_critic_func[name] + param.grad).detach()
            param.grad = -delta.detach().squeeze() * (z_critic_func[name] + param.grad)

        # Actor Update
        actor_optim.zero_grad()
        z_actor_func = {}
        for param in z_actor:
            z_actor_func[param] = (gamma * lambda_actor * z_actor[param]).detach()
        _, log_prob, _, _, _ = actor.sample(state, h_update)
        log_prob = torch.sum(log_prob.squeeze())
        log_prob.backward()
        for name, param in actor.named_parameters():
            z_actor[name] = (z_actor_func[name] + I * param.grad).detach()
            param.grad = -delta.detach().squeeze() * (z_actor_func[name] + I * param.grad)

        I = gamma * I

        actor_optim.step()
        critic_optim.step()

        return I, z_critic, z_actor


class Off_Policy_Agent(Agent_Trainer):
    def __init__(self,
                env,
                seed,
                inp_dim,
                hid_dim,
                action_dim,
                optimizer_spec_actor,
                optimizer_spec_critic,
                replay_buffer_size,
                batch_size,
                alpha,
                gamma,
                automatic_entropy_tuning, 
                learning_starts,
                learning_freq,
                save_iter,
                log_steps,
                frame_skips,
                model_save_path,
                reward_save_path,
                steps_save_path,
                action_scale,
                action_bias,
                policy_type,
                update_iters):

        super().__init__(env,
                        seed,
                        inp_dim,
                        hid_dim,
                        action_dim,
                        optimizer_spec_actor,
                        optimizer_spec_critic,
                        replay_buffer_size,
                        batch_size,
                        alpha,
                        gamma,
                        automatic_entropy_tuning, 
                        learning_starts,
                        learning_freq,
                        save_iter,
                        log_steps,
                        frame_skips,
                        model_save_path,
                        reward_save_path,
                        steps_save_path,
                        action_scale,
                        action_bias,
                        policy_type,
                        update_iters)
    
    def train(self, max_steps):

        actor_bg = Actor(self.inp_dim, self.hid_dim, self.action_dim, self.action_scale, self.action_bias).cuda()
        critic_bg = Critic(self.action_dim + self.inp_dim, self.hid_dim).cuda()
        critic_target_bg = Critic(self.action_dim + self.inp_dim, self.hid_dim).cuda()
        self.hard_update(critic_target_bg, critic_bg)

        actor_bg_optimizer = self.optimizer_spec_actor.constructor(actor_bg.parameters(), **self.optimizer_spec_actor.kwargs)
        critic_bg_optimizer = self.optimizer_spec_critic.constructor(critic_bg.parameters(), **self.optimizer_spec_critic.kwargs)

        target_entropy = -self.env.action_space.shape[0]
        log_alpha = torch.zeros(1, requires_grad=True, device="cuda:0")
        alpha_optim = optim.Adam([log_alpha], lr=.0003)

        policy_memory = ReplayBuffer(self.replay_buffer_size, self.seed)

        Statistics = {
            "mean_episode_rewards": [],
            "mean_episode_steps": [],
            "best_mean_episode_rewards": []
        }

        episode_reward = 0
        best_mean_episode_reward = -float("inf")
        episode_steps = 0
        total_episodes = 0
        avg_reward = [0]
        avg_steps = [0]

        ### GET INITAL STATE + RESET MODEL BY POSE
        state = self.env.reset(0)
        ep_trajectory = []

        #num_layers specified in the policy model 
        h_prev = torch.zeros(size=(1, 1, self.hid_dim), device="cuda")

        ### STEPS PER EPISODE ###
        for t in range(max_steps):

            with torch.no_grad():
                action, h_current = self.select_action(actor_bg, state, h_prev, evaluate=False)  # Sample action from policy

            ### TRACKING REWARD + EXPERIENCE TUPLE###
            for _ in range(self.frame_skips):
                next_state, reward, done = self.env.step(episode_steps, action, total_episodes)
                episode_steps += 1
                episode_reward += reward
                if done == True:
                    break

            mask = 1.0 if episode_steps == self.env.max_timesteps else float(not done)

            ep_trajectory.append((state, action, reward, next_state, mask))

            state = next_state
            h_prev = h_current

            ### EARLY TERMINATION OF EPISODE
            if done:

                total_episodes += 1

                # Add stats to lists
                avg_steps.append(episode_steps)
                avg_reward.append(episode_reward)

                # Push the episode to replay
                policy_memory.push(ep_trajectory)

                # reset training conditions
                h_prev = torch.zeros(size=(1, 1, self.hid_dim), device="cuda")
                state = self.env.reset(total_episodes) 

                # resest lists
                ep_trajectory = []

                ### 4. Log progress and keep track of statistics
                if len(avg_reward) > 0:
                    mean_episode_reward = np.mean(np.array(avg_reward)[-1000:])
                if len(avg_steps) > 0:
                    mean_episode_steps = np.mean(np.array(avg_steps)[-1000:])
                if len(avg_reward) > 10:
                    if mean_episode_reward > best_mean_episode_reward:
                        torch.save({
                            'iteration': t,
                            'agent_state_dict': actor_bg.state_dict(),
                            'critic_state_dict': critic_bg.state_dict(),
                            'agent_optimizer_state_dict': actor_bg_optimizer.state_dict(),
                            'critic_optimizer_state_dict': critic_bg_optimizer.state_dict(),
                        }, self.model_save_path + '.pth')

                    best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)

                Statistics["mean_episode_rewards"].append(mean_episode_reward)
                Statistics["mean_episode_steps"].append(mean_episode_steps)
                Statistics["best_mean_episode_rewards"].append(best_mean_episode_reward)

                print("Episode %d" % (total_episodes,))
                print("reward: %f" % episode_reward)
                print("steps: %f" % episode_steps)
                print("best mean reward: %f" % best_mean_episode_reward)
                sys.stdout.flush()

                if total_episodes % self.log_steps == 0:
                    # Dump statistics to pickle
                    np.save(f'{self.reward_save_path}.npy', Statistics["mean_episode_rewards"])
                    np.save(f'{self.steps_save_path}.npy', Statistics["mean_episode_steps"])
                    print("Saved to %s" % 'training_reports')
                
                # reset tracking variables
                episode_steps = 0
                episode_reward = 0

            # Apply Basal Ganglia update (using SAC)
            if total_episodes > self.learning_starts and total_episodes % self.learning_freq == 0 and len(policy_memory.buffer) > self.batch_size:

                for it in range(self.update_iters):
                    policy_loss, critic_loss = self.update(actor_bg,
                                                            critic_bg,
                                                            critic_target_bg,
                                                            critic_bg_optimizer,
                                                            actor_bg_optimizer,
                                                            policy_memory,
                                                            self.batch_size,
                                                            self.hid_dim,
                                                            self.gamma,
                                                            self.automatic_entropy_tuning,
                                                            log_alpha,
                                                            target_entropy,
                                                            self.alpha,
                                                            alpha_optim,
                                                            self.striatum_data,
                                                            self.policy_type)

    def update(self,
                actor,
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
                striatum_data,
                policy_type
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

        h_train = torch.zeros(size=(1, batch_size, hid_dim), device="cuda")

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

        pi_action_bat, log_prob_bat, _, pi_act, _ = actor.sample(state_batch, h_train, sampling= False, len_seq=len_seq)
        qf1_pi, qf2_pi = critic(state_batch, pi_action_bat, h_train, len_seq)

        qf1_pi = qf1_pi * loss_mask
        qf2_pi = qf2_pi * loss_mask
        log_prob_bat = log_prob_bat * loss_mask

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        policy_loss = ((alpha * log_prob_bat) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        if policy_type == "constrained":
            striatum_data = torch.tensor(striatum_data, device="cuda", dtype=torch.float32).repeat(batch_size, 1, 1)
            policy_loss = policy_loss + F.mse_loss(pi_act, striatum_data[:, 100:100+pi_act.shape[1], :])

        actor_optimizer.zero_grad()
        policy_loss.backward()

        d_act = torch.mean(torch.pow(pi_act * (1 - pi_act), 2), dim=(1, 0))
        actor.weight_l0_hh.grad += (1e-3 * actor.weight_l0_hh * d_act)

        torch.nn.utils.clip_grad_norm_(actor.parameters(), 1)
        actor_optimizer.step()

        if automatic_entropy_tuning:
            alpha_loss = -(log_alpha * (log_prob_bat + target_entropy).detach()).mean()

            alpha_optim.zero_grad()
            alpha_loss.backward()
            alpha_optim.step()

        self.soft_update(critic_target, critic, .005)

        return policy_loss.item(), qf_loss.item()
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)