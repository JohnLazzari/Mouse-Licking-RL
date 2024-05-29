import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from agent_learning import OptimizerSpec, On_Policy_Agent, Off_Policy_Agent
from utils.gym import get_env, get_wrapper_by_name
from lick_env import Lick_Env_Cont
import torch
import config
from utils.custom_optim import CustomAdamOptimizer

def main():

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    ### CREATE ENVIRONMENT ###
    torch.manual_seed(args.seed)
    env = Lick_Env_Cont(args.action_dim, args.timesteps, args.thresh, args.dt, args.beta, args.bg_scale, args.trajectory, args.full_alm_path, args.alm_hid_units)

    ### RUN TRAINING ###
    env = get_env(env, args.seed)

    ### OPTIMIZERS ###
    optimizer_spec_actor = OptimizerSpec(
        constructor=optim.AdamW,
        kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
    )

    optimizer_spec_critic = OptimizerSpec(
        constructor=optim.AdamW,
        kwargs=dict(lr=args.lr),
    )

    if args.update_method == "one_step":
        rl_setup = On_Policy_Agent(env,
                                    args.seed,
                                    args.inp_dim,
                                    args.hidden_dim,
                                    args.action_dim,
                                    optimizer_spec_actor,
                                    optimizer_spec_critic,
                                    args.policy_replay_size,
                                    args.policy_batch_size,
                                    args.alpha,
                                    args.gamma,
                                    args.automatic_entropy_tuning,
                                    args.learning_starts,
                                    args.learning_freq,
                                    args.save_iter,
                                    args.log_steps,
                                    args.frame_skips,
                                    args.model_save_path,
                                    args.reward_save_path,
                                    args.steps_save_path,
                                    args.action_scale,
                                    args.action_bias,
                                    args.policy_type,
                                    args.update_iters)

        rl_setup.train(args.max_steps)

    elif args.update_method == "sac":
        rl_setup = Off_Policy_Agent(env,
                                    args.seed,
                                    args.inp_dim,
                                    args.hidden_dim,
                                    args.action_dim,
                                    optimizer_spec_actor,
                                    optimizer_spec_critic,
                                    args.policy_replay_size,
                                    args.policy_batch_size,
                                    args.alpha,
                                    args.gamma,
                                    args.automatic_entropy_tuning,
                                    args.learning_starts,
                                    args.learning_freq,
                                    args.save_iter,
                                    args.log_steps,
                                    args.frame_skips,
                                    args.model_save_path,
                                    args.reward_save_path,
                                    args.steps_save_path,
                                    args.action_scale,
                                    args.action_bias,
                                    args.policy_type,
                                    args.update_iters)

        rl_setup.train(args.max_steps)

if __name__ == '__main__':
    main()
