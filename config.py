import configargparse

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument("--config", is_config_file=True, help="config file path")

    # General Training Parameters
    parser.add_argument('--gamma', 
                        type=float, 
                        default=0.99, 
                        help='discount factor for reward (default: 0.99)')

    parser.add_argument('--tau', 
                        type=float, 
                        default=0.005, 
                        help='target smoothing coefficient(τ) (default: 0.005)')

    parser.add_argument('--lr', 
                        type=float, 
                        default=0.0003, 
                        help='learning rate (default: 0.001)')

    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=0, 
                        help='weight decay parameter')

    parser.add_argument('--eps', 
                        type=float, 
                        default=0.01, 
                        help='optimizer epsilon')

    parser.add_argument('--learning_freq', 
                        type=int, 
                        default=1, 
                        help='mod episodes for gradient step')

    parser.add_argument('--learning_starts', 
                        type=int, 
                        default=100, 
                        help='episode count in which learning starts')

    parser.add_argument('--alpha', 
                        type=float, 
                        default=0.2, 
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')

    parser.add_argument('--automatic_entropy_tuning', 
                        type=bool, 
                        default=False, 
                        help='Automaically adjust α (default: False)')

    parser.add_argument('--seed', 
                        type=int, 
                        default=0, 
                        help='random seed (default: 123456)')

    parser.add_argument('--policy_batch_size', 
                        type=int, 
                        default=6, 
                        help='batch size (default: 6)')

    parser.add_argument('--hidden_dim', 
                        type=int, 
                        default=32, 
                        help='hidden size')

    parser.add_argument('--action_dim', 
                        type=int, 
                        default=1, 
                        help='action size')

    parser.add_argument('--inp_dim', 
                        type=int, 
                        default=3, 
                        help='state size')

    parser.add_argument('--log_steps', 
                        type=int, 
                        default=10, 
                        help='episodes before logging stats')

    parser.add_argument('--policy_replay_size', 
                        type=int, 
                        default=50000, 
                        help='size of replay buffer (default: 2800)')

    parser.add_argument('--save_iter', 
                        type=int, 
                        default=10000, 
                        help='number of episodes until checkpoint is saved')

    parser.add_argument('--thresh', 
                        type=int, 
                        default=1, 
                        help='threshold for alm to reach')

    parser.add_argument('--dt', 
                        type=float, 
                        default=0.1, 
                        help='dt of environment')

    parser.add_argument('--timesteps', 
                        type=int, 
                        default=60, 
                        help='number of timesteps for single episode (num / dt)')

    # Saving Parameters
    parser.add_argument('--model_save_path', 
                        type=str, 
                        default='',
                        help='path to folder and file name of model to save (do not put extension pth)')

    parser.add_argument('--reward_save_path', 
                        type=str, 
                        default='',
                        help='path to folder and file name to save rewards (do not put extension .npy)')

    parser.add_argument('--steps_save_path', 
                        type=str, 
                        help='path to folder and file name to save episode steps (do not put extension .npy)')

    return parser