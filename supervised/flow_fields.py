import numpy as np
import matplotlib.pyplot as plt
import torch
from models import RNN
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio

def NormalizeData(data, min, max):
    return (data - min) / (max - min)

class FlowFields():
    def __init__(self, inp_type, dimensions=2):
        
        self.inp_type = inp_type
        self.dimensions = dimensions
        self.x_pca = PCA(n_components=dimensions)
    
    def gather_inp_data(self, num_conditions, psth_folder=None, inp_region=None):

        x_inp = {}

        for cond in range(num_conditions):

            if self.inp_type == "linear":
                ramp = torch.linspace(0, 1, int((1.1 + (.3*cond)) / 0.01), dtype=torch.float32).unsqueeze(1)
                baseline = torch.zeros(size=(100, 1))
                x_inp[cond] = torch.cat((baseline, ramp), dim=0)
            elif self.inp_type == "psth":
                x_inp[cond] = sio.loadmat(f'{psth_folder}/{inp_region}_PSTH_cond{cond+1}.mat')['psth']
                min_data, max_data = np.min(x_inp[cond]), np.max(x_inp[cond])
                x_inp[cond] = torch.tensor(NormalizeData(np.squeeze(x_inp[cond]), min_data, max_data), dtype=torch.float32).unsqueeze(-1)

        x_inp_total = pad_sequence([x_inp[0], x_inp[1], x_inp[2]], batch_first=True)
        cue_inp = torch.zeros(size=(3, x_inp_total.shape[1], 1))
        cue_inp[:, 99, :] = 1
        total_inp = torch.cat((x_inp_total, cue_inp), dim=-1)
        
        return total_inp

    def generate_grid(self, num_points, lower_bound=0, upper_bound=1):

        # Num points is along each axis, not in total
        x = np.linspace(lower_bound, upper_bound, num_points)
        y = np.linspace(lower_bound, upper_bound, num_points)

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

def main():
    
    # General params
    inp_dim = 2
    hid_dim = 2
    out_dim = 1
    num_points = 100
    condition = 0 # 0, 1, or 2
    num_timepoints = 210 # 210, 240, or 270 (or 300 for data driven networks)
    perturbation = "striatum"

    # Saving and Loading params
    check_path = "checkpoints/rnn_goal_data_2n_delay.pth"
    save_name = "results/flow_fields/rnn_goal_data_delay/rnn_goal_data_2n_delay_flow"
    checkpoint = torch.load(check_path)
    
    # Create RNN
    rnn = RNN(inp_dim, hid_dim, out_dim)
    rnn.load_state_dict(checkpoint)

    # Gather data
    flow_field = FlowFields(inp_type="linear")

    x_inp = flow_field.gather_inp_data(3, psth_folder="data/PCs_PSTH", inp_region="striatum")
    x, y, data_coords = flow_field.generate_grid(num_points)

    # Sample many hidden states to get pcs for dimensionality reduction
    h = torch.rand(size=(1, 10000, hid_dim))
    sampled_acts = []
    for t in range(x_inp.shape[1]):
        with torch.no_grad():
            cur_inp = x_inp[condition:condition+1, t:t+1, :].repeat(10000, 1, 1)
            out, _, act = rnn(cur_inp, h)
            sampled_acts.append(act)
    sampled_acts = torch.cat(sampled_acts, dim=1)
    sampled_acts = torch.reshape(sampled_acts, shape=(sampled_acts.shape[0] * sampled_acts.shape[1], sampled_acts.shape[2]))

    # Get the typical trajectory throughout the task
    h = torch.zeros(size=(1, 1, hid_dim))
    act_cond = []
    inp_idx = 0
    for t in range(x_inp.shape[1]):
        with torch.no_grad():
            cur_inp = x_inp[condition:condition+1, inp_idx:inp_idx+1, :]
            if perturbation == "striatum" and t > 135 and t < 165:
                cur_inp = torch.zeros_like(cur_inp)
                inp_idx = 100
            out, h, act = rnn(cur_inp, h)
            if perturbation == "alm" and t > 110 and t < 140:
                h = torch.zeros_like(h)
        act_cond.append(h.numpy().squeeze(0))
        inp_idx += 1
    act_cond = np.concatenate(act_cond, axis=0)

    if hid_dim > 2:
        flow_field.fit_pca(sampled_acts)
        grid = flow_field.inverse_pca(data_coords)
        grid = torch.tensor(grid, dtype=torch.float32)
    else:
        grid = data_coords

    # initialize activity dict
    next_acts = {}
    for i in range(num_timepoints):
        next_acts[i] = []

    # Go through activities and generate h_t+1
    inp_idx = 0
    for t in range(num_timepoints):
        print(f"input number: {t}")
        cur_x_inp = x_inp[condition, inp_idx, :].unsqueeze(0).unsqueeze(0)
        if perturbation == "striatum" and t > 135 and t < 165:
            cur_x_inp = torch.zeros_like(cur_x_inp)
            inp_idx = 100
        for h_0 in grid:
            with torch.no_grad():
                h_0 = h_0.unsqueeze(0).unsqueeze(0)
                _, _, act = rnn(cur_x_inp, h_0)
                next_acts[t].append(act[0, 0, :].numpy())
        inp_idx += 1

    # Reshape data back to grid
    data_coords = data_coords.numpy()
    data_coords = np.reshape(data_coords, (num_points, num_points, data_coords.shape[-1]))
    for i in range(num_timepoints):
        next_acts[i] = np.array(next_acts[i])
        if hid_dim > 2:
            next_acts[i] = flow_field.transform_pca(next_acts[i])
        next_acts[i] = np.reshape(next_acts[i], (num_points, num_points, next_acts[i].shape[-1]))
    
    x_vels = {}
    y_vels = {}
    for i in range(num_timepoints):
        x_vels[i] = next_acts[i][:, :, 0] - data_coords[:, :, 0]
        y_vels[i] = next_acts[i][:, :, 1] - data_coords[:, :, 1]

    # Reduce original trajectory
    if hid_dim > 2:
        act_cond = flow_field.transform_pca(act_cond.squeeze().numpy())
    
    for i in range(num_timepoints):
        plt.streamplot(x, y, x_vels[i], y_vels[i], color="black")
        plt.scatter(act_cond[int(i * 1), 0], act_cond[int(i * 1), 1], c="blue", edgecolors="face", s=150)
        plt.yticks([])
        plt.xticks([])
        plt.savefig(save_name + f"_perturbation_{perturbation}" + f"_cond{condition}_inp{i}.png")
        plt.close()
    
if __name__ == "__main__":
    main()