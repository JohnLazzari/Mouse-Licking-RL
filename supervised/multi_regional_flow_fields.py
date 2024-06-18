import numpy as np
import matplotlib.pyplot as plt
import torch
from models import RNN_MultiRegional
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_delay_data

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.04)
LR = 1e-4
DT = 1e-3
CONDITION = 0
NUM_POINTS = 10000
MODEL_TYPE = "constraint" # constraint, no_constraint, no_constraint_thal
PERTURBATION = "striatum"
CHECK_PATH = "checkpoints/rnn_goal_data_2n_delay.pth"
SAVE_NAME = "results/flow_fields/rnn_goal_data_delay/rnn_goal_data_2n_delay_flow"

class FlowFields():
    def __init__(self, inp_type, dimensions=2):
        
        self.inp_type = inp_type
        self.dimensions = dimensions
        self.x_pca = PCA(n_components=dimensions)
    
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
    
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM)
    rnn.load_state_dict(checkpoint)

    # Gather data
    flow_field = FlowFields(inp_type="linear")

    # Get input and output data
    x_data, y_data, len_seq = gather_delay_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    y_data = y_data.cuda()

    x, y, data_coords = flow_field.generate_grid(num_points=NUM_POINTS)

    # Sample many hidden states to get pcs for dimensionality reduction
    h = torch.rand(size=(1, 1, HID_DIM))
    sampled_acts = []
    for t in range(len_seq[CONDITION]):
        with torch.no_grad():
            cur_inp = x_data[CONDITION:CONDITION+1, t:t+1, :]
            _, _, act, _, _ = rnn(x_data, h, x, 0, noise=False)
            sampled_acts.append(act)
    sampled_acts = torch.cat(sampled_acts, dim=1)
    sampled_acts = torch.reshape(sampled_acts, shape=(sampled_acts.shape[0] * sampled_acts.shape[1], sampled_acts.shape[2]))

    flow_field.fit_pca(sampled_acts)
    grid = flow_field.inverse_pca(data_coords)
    grid = torch.tensor(grid, dtype=torch.float32)

    # initialize activity dict
    next_acts = {}
    for i in range(NUM_TIMEPOINTS):
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