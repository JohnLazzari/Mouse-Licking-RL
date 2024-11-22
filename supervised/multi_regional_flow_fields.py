import numpy as np
import matplotlib.pyplot as plt
import torch
from models import CBGTCL
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_inp_data, load_variables_from_file, get_ramp, get_data, get_masks, gather_train_val_test_split
from FixedPointFinderTorch import FixedPointFinderTorch
import torch.nn.functional as F

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

CONDITION = 1
NUM_POINTS = 100
START_REGION = "striatum" # str, alm, str2thal, or snr
END_REGION = "thal"
TIME_SKIPS = 10
PERTURBATION = False
PERTURBED_REGION = "alm" # str or alm
ITI_STEPS = 100
START_SILENCE = 160
END_SILENCE = 220
LOWER_BOUND_X = -25
UPPER_BOUND_X = 25
LOWER_BOUND_Y = -10
UPPER_BOUND_Y = 10
MODEL_NAMES = ["22_11_2024_11_58_38"]
SPECIFICATIONS_PATH = "checkpoints/model_specifications/"
CHECK_PATH = "checkpoints/"
SAVE_NAME_PATH = "results/flow_fields/"
DATA_SPLIT = "training"

class FlowFields():
    def __init__(self, dimensions=2):
        
        self.dimensions = dimensions
        self.x_pca = PCA(n_components=dimensions)
    
    def generate_grid(
        self, 
        num_points, 
        lower_bound_x=-12, 
        upper_bound_x=18,
        lower_bound_y=-12, 
        upper_bound_y=18
    ):

        # Num points is along each axis, not in total
        x = np.linspace(lower_bound_x, upper_bound_x, num_points)
        y = np.linspace(lower_bound_y, upper_bound_y, num_points)

        xv, yv = np.meshgrid(x, y)
        xv = np.expand_dims(xv, axis=-1)
        yv = np.expand_dims(yv, axis=-1)

        coordinates = np.concatenate((xv, yv), axis=-1)
        coordinates_data = torch.tensor(coordinates, dtype=torch.float32)
        coordinates_data = torch.flatten(coordinates_data, start_dim=0, end_dim=1)

        return x, y, coordinates_data

    def fit_pca(self, x):
        self.x_pca.fit(x)

    def transform_pca(self, x):
        transformed = self.x_pca.transform(x)
        return transformed
    
    def inverse_pca(self, x):
        inv_transformed = self.x_pca.inverse_transform(x)
        return inv_transformed
    
    def reset_pca(self):
        self.x_pca = PCA(n_components=self.dimensions)


def gen_inverse_grid(mrnn, start_region, end_region, data_coords, flow_field, iti_inp, cue_inp):

    batch_size = iti_inp.shape[0]
    seq_len = iti_inp.shape[1]

    # Initialize hidden states
    hn = torch.zeros(size=(1, batch_size, mrnn.total_num_units), device="cuda")
    xn = torch.zeros(size=(1, batch_size, mrnn.total_num_units), device="cuda")
    inhib_stim = torch.zeros(size=(batch_size, seq_len, mrnn.total_num_units), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        act = mrnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=False)

    # Reshape before PCA
    region_act = []
    start_reached = False
    for region in mrnn.region_dict:
        if region == start_region:
            start_reached = True
        if start_reached:
            region_act.append(mrnn.get_region_activity(region, act))
        if region == end_region:
            break

    region_act = torch.cat(region_act, dim=-1)
    region_act = torch.reshape(region_act, shape=(region_act.shape[0] * region_act.shape[1], region_act.shape[2])) 
    region_act = region_act.detach().cpu().numpy()

    # Do PCA on the specified region
    flow_field.fit_pca(region_act[:seq_len, :])

    # Inverse PCA to input grid into network
    grid = flow_field.inverse_pca(data_coords)
    grid = grid.clone().detach().cuda()

    return grid, act


def main():
    
    # Loop through each model specified
    for model in MODEL_NAMES:

        # Open json for mRNN configuration file
        config_name = SPECIFICATIONS_PATH + model + ".json"
        
        # Load in variables from the training specs txt file
        train_specs = load_variables_from_file(SPECIFICATIONS_PATH + model + ".txt")

        # Unload variables
        inp_dim = train_specs["inp_dim"]
        out_dim = train_specs["out_dim"]
        epochs = train_specs["epochs"]
        lr = train_specs["lr"]
        dt = train_specs["dt"]
        weight_decay = train_specs["weight_decay"]
        constrained = train_specs["constrained"]
        trial_epoch = train_specs["trial_epoch"]
        nmf = train_specs["nmf"]
        n_components = train_specs["n_components"]
        out_type = train_specs["out_type"]

        # Load saved model
        checkpoint = torch.load(CHECK_PATH + model + ".pth")
        rnn = CBGTCL(config_name, inp_dim, out_dim, out_type, constrained=constrained).cuda()
        rnn.load_state_dict(checkpoint)

        # Get ramping activity
        if out_type == "ramp":

            neural_act, peak_times = get_ramp(dt=dt)
            neural_act = neural_act.cuda()

        elif out_type == "data":

            neural_act, peak_times = get_data(
                dt,
                trial_epoch,
                nmf,
                n_components
            )

            neural_act = neural_act.cuda()

        # Get input and output data
        iti_inp, cue_inp, len_seq = gather_inp_data(inp_dim=inp_dim, peaks=peak_times)
        iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()
        loss_mask = get_masks(out_dim, len_seq)

        data_dict = gather_train_val_test_split(neural_act, peak_times, iti_inp, cue_inp, loss_mask)

        # Gather data
        flow_field = FlowFields()

        # Generate a grid which represents combinations of values of the first two PCs of the network
        x, y, data_coords = flow_field.generate_grid(
            num_points=NUM_POINTS,
            lower_bound_x=LOWER_BOUND_X,
            upper_bound_x=UPPER_BOUND_X,
            lower_bound_y=LOWER_BOUND_Y,
            upper_bound_y=UPPER_BOUND_Y
        )

        grid, act = gen_inverse_grid(
            rnn.mrnn, 
            START_REGION, 
            END_REGION,
            data_coords, 
            flow_field, 
            data_dict[DATA_SPLIT]["iti_inp"][CONDITION:CONDITION+1, :, :], 
            data_dict[DATA_SPLIT]["cue_inp"][CONDITION:CONDITION+1, :, :]
        )

        # Initialize activity dict
        next_acts = {}
        # Repeat over the sequence length dimension to match activity
        grid = grid.unsqueeze(0).repeat(act.shape[1], 1, 1)
        # Repeat along the batch dimension to match the grid
        full_act_batch = act[CONDITION:CONDITION+1, :, :].repeat(grid.shape[0], 1, 1)

        # Gather input
        x_0_flow = []
        inhib_stim = torch.zeros(size=(grid.shape[0], 1, rnn.total_num_units), device="cuda")
        start_reached = False
        end_reached = False
        for region in rnn.mrnn.region_dict:
            if region == START_REGION:
                x_0_flow.append(grid)
                start_reached = True
            if not start_reached:
                x_0_flow.append(rnn.mrnn.get_region_activity(region, full_act_batch))
            if start_reached and end_reached:
                x_0_flow.append(rnn.mrnn.get_region_activity(region, full_act_batch))
            if region == END_REGION:
                end_reached = True
        x_0_flow = torch.cat(x_0_flow, dim=-1)
        h_0_flow = F.relu(x_0_flow)
                
                
                


            




            with torch.no_grad():

                if REGION == "str":

                    h_0 = F.relu(torch.cat([grid, sampled_acts[0, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0))
                    x_0 = torch.cat([grid, sampled_acts[0, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0)

                elif REGION == "str2thal":

                    h_0 = F.relu(torch.cat([grid, sampled_acts[0, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0))
                    x_0 = torch.cat([grid, sampled_acts[0, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0)

                elif REGION == "alm":
                    
                    h_0 = F.relu(torch.cat([sampled_acts[0, t:t+1, :region_start].repeat(grid.shape[0], 1), grid], dim=1).unsqueeze(0))
                    x_0 = torch.cat([sampled_acts[0, t:t+1, :region_start].repeat(grid.shape[0], 1), grid], dim=1).unsqueeze(0)

                elif REGION == "snr":

                    h_0 = F.relu(torch.cat([sampled_acts[0, t:t+1, :region_start].repeat(grid.shape[0], 1), grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0))
                    x_0 = torch.cat([sampled_acts[0, t:t+1, :region_start].repeat(grid.shape[0], 1), grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0)

                cur_iti_inp = iti_inp[CONDITION:CONDITION+1, t:t+1, :]
                cur_cue_inp = cue_inp[CONDITION:CONDITION+1, t:t+1, :]

                _, _, act = rnn(cur_iti_inp, cur_cue_inp, h_0, x_0, inhib_stim, noise=False)

                next_acts[t] = act[:, 0, region_start:region_end].detach().cpu().numpy()
            
            act_max, act_min = torch.max(h_0), torch.min(h_0)
            print("Max: ", act_max)
            print("Min: ", act_min)

        # Reshape data back to grid
        data_coords = data_coords.numpy()
        data_coords = np.reshape(data_coords, (NUM_POINTS, NUM_POINTS, data_coords.shape[-1]))

        sampled_acts_region = sampled_acts[0, :, region_start:region_end].detach().cpu().numpy()
        sampled_acts_region = flow_field.transform_pca(sampled_acts_region)

        for t in range(50, len_seq[CONDITION], TIME_SKIPS):

            next_acts[t] = flow_field.transform_pca(np.array(next_acts[t]))
            next_acts[t] = np.reshape(next_acts[t], (NUM_POINTS, NUM_POINTS, next_acts[t].shape[-1]))

        x_vels = {}
        y_vels = {}

        for i in range(50, len_seq[CONDITION], TIME_SKIPS):

            x_vels[i] = next_acts[i][:, :, 0] - data_coords[:, :, 0]
            y_vels[i] = next_acts[i][:, :, 1] - data_coords[:, :, 1]
        
        speeds = {}
        
        for i in range(50, len_seq[CONDITION], TIME_SKIPS):
            speed = np.sqrt(x_vels[i]**2 + y_vels[i]**2)
            c = speed / speed.max()
            speeds[i] = c

    for i in range(100, len_seq[CONDITION], TIME_SKIPS):
        
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(data_coords[:, :, 0], data_coords[:, :, 1], speeds[i][:, :], cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)
        plt.tight_layout()
        plt.savefig(SAVE_NAME + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}_energy.png")
        plt.close()
        
        plt.scatter(sampled_acts_region[i, 0], sampled_acts_region[i, 1], c="red", s=250, zorder=10)
        plt.plot(sampled_acts_region[100:i, 0], sampled_acts_region[100:i, 1], c="blue", linewidth=4, zorder=5)
        plt.streamplot(x, y, x_vels[i], y_vels[i], color=speeds[i], cmap="gray", linewidth=3, arrowsize=2, zorder=0)
        plt.yticks([])
        plt.xticks([])

        plt.tight_layout()
        plt.savefig(SAVE_NAME + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.png")
        plt.savefig(SAVE_NAME_EPS + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.eps")
        plt.close()
    
if __name__ == "__main__":
    main()