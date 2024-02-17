import gym
import torch.optim as optim
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

from sac_model import Critic, Actor_Seq
from sac_learn import OptimizerSpec, sac_learn
from utils.gym import get_env, get_wrapper_by_name
from lick_env import Lick_Env_Cont, Kinematics_Env, Kinematics_Jaw_Env
from utils.custom_optim import CustomAdamOptimizer
import torch
import config

def main():

    torch.autograd.set_detect_anomaly(True)

    ### PARAMETERS ###
    parser = config.config_parser()
    args = parser.parse_args()

    ### CREATE ENVIRONMENT ###
    torch.manual_seed(args.seed)
    
    if args.env == "kinematics":
        env = Kinematics_Env(args.action_dim, args.dt, args.kinematics_folder)
    elif args.env == "kinematics_jaw":
        env = Kinematics_Jaw_Env(args.action_dim, args.dt, args.kinematics_folder)
    elif args.env == "lick_ramp":
        env = Lick_Env_Cont(args.action_dim, args.timesteps, args.thresh, args.dt, args.beta, args.bg_scale, args.alm_data_path)

    ### RUN TRAINING ###
    env = get_env(env, args.seed)

    agent = Actor_Seq
    actor_optimizer_spec = OptimizerSpec(
        constructor=CustomAdamOptimizer,
        kwargs=dict(lr=args.lr),
    )
    critic_optimizer_spec = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=args.lr),
    )

    sac_learn(
        env,
        args.seed,
        args.inp_dim,
        args.hidden_dim,
        args.action_dim,
        agent,
        Critic,
        actor_optimizer_spec,
        critic_optimizer_spec,
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
    )

if __name__ == '__main__':
    main()
