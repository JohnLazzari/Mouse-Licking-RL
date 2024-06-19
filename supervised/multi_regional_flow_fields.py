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
NUM_POINTS = 100
MODEL_TYPE = "constraint" # constraint, no_constraint, no_constraint_thal
PERTURBATION = ""
CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
SAVE_NAME = "results/flow_fields/multi_regional_d1d2/multi_regional_d1d2_flow"

class FlowFields():
    def __init__(self, dimensions=2):
        
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
    rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)

    # Gather data
    flow_field = FlowFields()

    # Get input and output data
    x_data, y_data, len_seq = gather_delay_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()
    y_data = y_data.cuda()

    x, y, data_coords = flow_field.generate_grid(num_points=NUM_POINTS)

    # Sample many hidden states to get pcs for dimensionality reduction
    hn = torch.rand(size=(1, 1, HID_DIM*6)).cuda()
    xn = hn

    sampled_acts = []
    for t in range(len_seq[CONDITION]):
        with torch.no_grad():
            cur_inp = x_data[CONDITION:CONDITION+1, t:t+1, :]
            _, hn, act, _, _ = rnn(cur_inp, hn, xn, 0, noise=False)
            sampled_acts.append(act)
    sampled_acts = torch.cat(sampled_acts, dim=1)
    sampled_acts = torch.reshape(sampled_acts, shape=(sampled_acts.shape[0] * sampled_acts.shape[1], sampled_acts.shape[2]))

    flow_field.fit_pca(sampled_acts.detach().cpu().numpy())
    grid = flow_field.inverse_pca(data_coords)
    grid = torch.tensor(grid, dtype=torch.float32)

    alm_mask = rnn.alm_mask
    str_mask = rnn.str_d1_mask
    ITI_steps = 1000
    extra_steps = 0
    start_silence = 600 + ITI_steps
    end_silence = 1100 + ITI_steps

    if PERTURBATION == "striatum" or PERTURBATION == "alm":
        len_seq_act = len_seq[CONDITION] + (end_silence - start_silence) + extra_steps
    else:
        len_seq_act = len_seq[CONDITION]

    # initialize activity dict
    next_acts = {}
    for i in range(len_seq_act):
        next_acts[i] = []

    # Go through activities and generate h_t+1
    for t in range(0, len_seq_act, 100):

        print(f"input number: {t}")

        if t < ITI_steps:
            inp = x_data[CONDITION:CONDITION+1, 0:1, :]
        else:
            inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]

        if PERTURBATION == "alm" and t > start_silence and t < end_silence:
            inhib_stim = -10 * alm_mask
            inp = 0*x_data[CONDITION:CONDITION+1, 0:1, :]
        elif PERTURBATION == "striatum" and t > start_silence and t < end_silence:
            inhib_stim = -0.25 * str_mask
            #inhib_stim = 1 * str_mask
        else:
            inhib_stim = 0

        for h_0 in grid:
            with torch.no_grad():
                h_0 = h_0.unsqueeze(0).unsqueeze(0).cuda()
                _, _, act, _, _ = rnn(inp, h_0, xn, inhib_stim, noise=False)
                next_acts[t].append(act[0, 0, :HID_DIM].detach().cpu().numpy())

    # Reshape data back to grid
    data_coords = data_coords.numpy()
    data_coords = np.reshape(data_coords, (NUM_POINTS, NUM_POINTS, data_coords.shape[-1]))
    for i in range(len_seq_act):
        next_acts[i] = flow_field.transform_pca(next_acts[i])
        next_acts[i] = np.reshape(next_acts[i], (NUM_POINTS, NUM_POINTS, next_acts[i].shape[-1]))
    
    x_vels = {}
    y_vels = {}
    for i in range(len_seq_act):
        x_vels[i] = next_acts[i][:, :, 0] - data_coords[:, :, 0]
        y_vels[i] = next_acts[i][:, :, 1] - data_coords[:, :, 1]

    # Reduce original trajectory
    act_cond = flow_field.transform_pca(act_cond.squeeze().numpy())
    
    for i in range(len_seq_act):
        plt.streamplot(x, y, x_vels[i], y_vels[i], color="black")
        plt.scatter(act_cond[int(i * 1), 0], act_cond[int(i * 1), 1], c="blue", edgecolors="face", s=150)
        plt.yticks([])
        plt.xticks([])
        plt.savefig(SAVE_NAME + f"_perturbation_{PERTURBATION}" + f"_cond{CONDITION}_inp{i}.png")
        plt.close()
    
if __name__ == "__main__":
    main()