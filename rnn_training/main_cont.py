import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sac_model import Actor, Critic
from sac_learn import OptimizerSpec, sac_learn
from utils.gym import get_env, get_wrapper_by_name
from lick_env import Lick_Env_Cont, Kinematics_Jaw_Env
import torch
import config
from utils.custom_optim import CustomAdamOptimizer

def main():

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    ### CREATE ENVIRONMENT ###
    torch.manual_seed(args.seed)
    
    if args.env == "kinematics_jaw":
        env = Kinematics_Jaw_Env(args.action_dim, args.dt, args.kinematics_folder, args.alm_data_path, args.bg_scale)
    elif args.env == "lick_ramp":
        env = Lick_Env_Cont(args.action_dim, args.timesteps, args.thresh, args.dt, args.beta, args.bg_scale, args.alm_data_path)

    ### RUN TRAINING ###
    env = get_env(env, args.seed)

    if args.model_type == "gru":
        optimizer_spec_actor = OptimizerSpec(
            constructor=optim.Adam,
            kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
        )
        optimizer_spec_critic = OptimizerSpec(
            constructor=optim.Adam,
            kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
        )
    elif args.model_type == "sparse":
        
        optimizer_spec_actor = OptimizerSpec(
            constructor=CustomAdamOptimizer,
            kwargs=None
        )
        optimizer_spec_critic = OptimizerSpec(
            constructor=optim.Adam,
            kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
        )

    sac_learn(
        env,
        args.seed,
        args.inp_dim,
        args.hidden_dim,
        args.action_dim,
        args.model_type,
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
        args.model_type
    )

if __name__ == '__main__':
    main()
