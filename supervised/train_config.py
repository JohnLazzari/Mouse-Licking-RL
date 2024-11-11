import configargparse
import argparse

def config_parser():
    parser = configargparse.ArgumentParser()
    
    parser.add_argument("--config", is_config_file=True, help="config file path")
    
    parser.add_argument('--out_dim', 
                        type=int, 
                        default=1,
                        help='dimension of output layer')

    parser.add_argument('--inp_dim', 
                        type=int, 
                        default=26, 
                        help='dimension of input')

    parser.add_argument('--epochs', 
                        type=int, 
                        default=5000, 
                        help='training iterations')

    parser.add_argument('--lr', 
                        type=float, 
                        default=1e-4, 
                        help='learning rate (default: 1e-4)')

    parser.add_argument('--dt', 
                        type=float, 
                        default=1e-2, 
                        help='change in time at each timestep (default: 1e-2)')

    parser.add_argument('--weight_decay', 
                        type=float, 
                        default=1e-3, 
                        help='weight decay value (default: 1e-3)')

    parser.add_argument('--constrained', action=configargparse.BooleanOptionalAction)

    parser.add_argument('--trial_epoch', 
                        type=str, 
                        default="delay", 
                        help='delay or full trial epoch (default: delay)')

    parser.add_argument('--nmf', action=configargparse.BooleanOptionalAction)

    parser.add_argument('--n_components', 
                        type=int, 
                        default=5, 
                        help='number of components for nmf (default: 5)')

    parser.add_argument('--out_type', 
                        type=str, 
                        default="ramp", 
                        help='ramp or data (default: ramp)')

    parser.add_argument('--save_path', 
                        type=str, 
                        default="checkpoints/d1d2", 
                        help='path to save network')

    parser.add_argument('--model_specifications_path', 
                        type=str, 
                        default="checkpoints/model_specifications/", 
                        help='path to save network')

    parser.add_argument('--mrnn_config_file', 
                        type=str,
                        default="supervised/configurations/mRNN_ramp.json",
                        help='path of configuration for mRNN')

    return parser