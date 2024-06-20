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
REGION = "str2thal" # str, alm, or str2thal
TIME_SKIPS = 500
PERTURBATION = False
PERTURBED_REGION = "alm" # str or alm
CHECK_PATH = "checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_d1d2.pth"
SAVE_NAME = "results/flow_fields/multi_regional_d1d2/multi_regional_d1d2_flow"

class FlowFields():
    def __init__(self, dimensions=2):
        
        self.dimensions = dimensions
        self.x_pca = PCA(n_components=dimensions)
    
    def generate_grid(self, num_points, lower_bound=-10, upper_bound=25):

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
    
    ITI_steps = 1000
    extra_steps = 0
    start_silence = 600 + ITI_steps
    end_silence = 1100 + ITI_steps
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional(INP_DIM, HID_DIM, OUT_DIM).cuda()
    rnn.load_state_dict(checkpoint)
    alm_mask = rnn.alm_mask
    str_mask = rnn.str_d1_mask

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


    # Get original trajectory
    for t in range(len_seq[CONDITION]):

        with torch.no_grad():

            cur_inp = x_data[CONDITION:CONDITION+1, t:t+1, :]
            _, hn, act, _, _ = rnn(cur_inp, hn, xn, 0, noise=False)
            sampled_acts.append(act)

    sampled_acts = torch.cat(sampled_acts, dim=1)
    sampled_acts = torch.reshape(sampled_acts, shape=(sampled_acts.shape[0] * sampled_acts.shape[1], sampled_acts.shape[2])) 
    sampled_acts = sampled_acts.detach().cpu().numpy()

    if REGION == "str":
        flow_field.fit_pca(sampled_acts[:, :int(HID_DIM/2)])
    elif REGION == "alm":
        flow_field.fit_pca(sampled_acts[:, HID_DIM*5:])
    elif REGION == "str2thal":
        flow_field.fit_pca(sampled_acts[:, :HID_DIM*5])

    grid = flow_field.inverse_pca(data_coords)
    grid = torch.tensor(grid, device="cuda").clone()

    if PERTURBATION == True:
        len_seq_act = len_seq[CONDITION] + (end_silence - start_silence) + extra_steps
    else:
        len_seq_act = len_seq[CONDITION]


    # Get perturbed trajectory
    perturbed_acts = []
    hn = torch.rand(size=(1, 1, HID_DIM*6)).cuda()
    xn = hn

    for t in range(len_seq_act):

        if t < ITI_steps:
            inp = x_data[CONDITION:CONDITION+1, 0:1, :]
        else:
            inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]

        with torch.no_grad():        

            if PERTURBED_REGION == "alm" and t > start_silence and t < end_silence:
                    inhib_stim = -10 * alm_mask
                    inp = 0*x_data[CONDITION:CONDITION+1, 0:1, :]
            elif PERTURBED_REGION == "str" and t > start_silence and t < end_silence:
                inhib_stim = -0.25 * str_mask
                #inhib_stim = 1 * str_mask
            else:
                inhib_stim = 0
                    
            _, hn, _, xn, _ = rnn(inp, hn, xn, inhib_stim, noise=False)
            perturbed_acts.append(hn)

    # initialize activity dict
    next_acts = {}
    for i in range(0, len_seq_act, TIME_SKIPS):
        next_acts[i] = []

    hn = torch.rand(size=(1, 1, HID_DIM*6)).cuda()
    xn = hn

    if PERTURBATION == True:
        hidden_input = torch.concatenate(perturbed_acts, dim=1).clone().cuda()
    else:
        hidden_input = torch.tensor(sampled_acts).clone().unsqueeze(0).cuda()

    # Go through activities and generate h_t+1
    for t in range(0, len_seq_act, TIME_SKIPS):

        print(f"input number: {t}")

        if t < ITI_steps:
            inp = x_data[CONDITION:CONDITION+1, 0:1, :]
        else:
            inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]
        
        if PERTURBATION == True and t > start_silence and t < end_silence:
            inp = 0 * x_data[CONDITION:CONDITION+1, 0:1, :]

        for h_0 in grid:

            with torch.no_grad():

                h_0 = h_0.unsqueeze(0).cuda()

                if REGION == "str":
                    h_0 = torch.cat([h_0, hidden_input[0, 0:1, int(HID_DIM/2):]], dim=1).unsqueeze(0)
                elif REGION == "str2thal":
                    h_0 = torch.cat([h_0, hidden_input[0, 0:1, HID_DIM*5:]], dim=1).unsqueeze(0)
                elif REGION == "alm":
                    h_0 = torch.cat([hidden_input[0, 0:1, :HID_DIM*5], h_0], dim=1).unsqueeze(0)

                _, _, act, _, _ = rnn(inp, h_0, xn, inhib_stim, noise=False)

                if REGION == "str": 
                    next_acts[t].append(act[0, 0, :int(HID_DIM/2)].detach().cpu().numpy())
                elif REGION == "str2thal": 
                    next_acts[t].append(act[0, 0, :HID_DIM*5].detach().cpu().numpy())
                elif REGION == "alm":
                    next_acts[t].append(act[0, 0, HID_DIM*5:].detach().cpu().numpy())

    # Reshape data back to grid
    data_coords = data_coords.numpy()
    data_coords = np.reshape(data_coords, (NUM_POINTS, NUM_POINTS, data_coords.shape[-1]))

    for i in range(0, len_seq_act, TIME_SKIPS):

        next_acts[i] = flow_field.transform_pca(np.array(next_acts[i]))
        next_acts[i] = np.reshape(next_acts[i], (NUM_POINTS, NUM_POINTS, next_acts[i].shape[-1]))
    
    if REGION == "str":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, :int(HID_DIM/2)])
    elif REGION == "alm":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, HID_DIM*5:])
    elif REGION == "str2thal":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, :HID_DIM*5])
    
    x_vels = {}
    y_vels = {}

    for i in range(0, len_seq_act, TIME_SKIPS):

        x_vels[i] = next_acts[i][:, :, 0] - data_coords[:, :, 0]
        y_vels[i] = next_acts[i][:, :, 1] - data_coords[:, :, 1]

    for i in range(0, len_seq_act, TIME_SKIPS):

        plt.streamplot(x, y, x_vels[i], y_vels[i], color="black")
        plt.scatter(sampled_acts_region[i, 0], sampled_acts_region[i, 1])
        plt.yticks([])
        plt.xticks([])
        plt.savefig(SAVE_NAME + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.png")
        plt.close()
    
if __name__ == "__main__":
    main()