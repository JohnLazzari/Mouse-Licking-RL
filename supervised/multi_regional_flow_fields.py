import numpy as np
import matplotlib.pyplot as plt
import torch
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_D1, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_inp_data

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1
INP_DIM = int(HID_DIM*0.1)
DT = 1e-3
CONDITION = 0
NUM_POINTS = 100
MODEL_TYPE = "d1d2" # constraint, no_constraint, no_constraint_thal
REGION = "str2thal" # str, alm, str2thal, or snr
TIME_SKIPS = 500
PERTURBATION = False
PERTURBED_REGION = "str" # str or alm
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}.pth"
SAVE_NAME = f"results/flow_fields/multi_regional_{MODEL_TYPE}/multi_regional_{MODEL_TYPE}_flow"
SAVE_NAME_EPS = f"results/flow_fields/multi_regional_{MODEL_TYPE}_eps/multi_regional_{MODEL_TYPE}_flow"

class FlowFields():
    def __init__(self, dimensions=2):
        
        self.dimensions = dimensions
        self.x_pca = PCA(n_components=dimensions)
    
    def generate_grid(self, num_points, lower_bound=-12, upper_bound=18):

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
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM).cuda()
    if MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM).cuda()
    if MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM).cuda()

    rnn.load_state_dict(checkpoint)

    alm_mask = rnn.alm_mask
    str_mask = rnn.str_d1_mask

    if MODEL_TYPE == "d1d2":

        total_num_units = HID_DIM * 6
        str_start = 0
        snr_start = HID_DIM*3
        thal_start = HID_DIM*4
        alm_start = HID_DIM*5

    elif MODEL_TYPE == "d1":

        total_num_units = HID_DIM * 4
        str_start = 0
        thal_start = HID_DIM*2
        alm_start = HID_DIM*3

    elif MODEL_TYPE == "stralm":

        total_num_units = HID_DIM * 2
        str_start = 0
        alm_start = HID_DIM

    # Gather data
    flow_field = FlowFields()

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    x, y, data_coords = flow_field.generate_grid(num_points=NUM_POINTS)

    hn = torch.zeros(size=(1, 3, total_num_units)).cuda()
    xn = hn

    inhib_stim = torch.zeros(size=(1, x_data.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, act, _, _ = rnn(x_data, hn, xn, inhib_stim, noise=False)

    sampled_acts = act[CONDITION:CONDITION+1, :len_seq[CONDITION], :]
    sampled_acts = torch.reshape(sampled_acts, shape=(sampled_acts.shape[0] * sampled_acts.shape[1], sampled_acts.shape[2])) 
    sampled_acts = sampled_acts.detach().cpu().numpy()

    if REGION == "str":
        flow_field.fit_pca(sampled_acts[:, str_start:str_start+HID_DIM])
    elif REGION == "alm":
        flow_field.fit_pca(sampled_acts[:, alm_start:alm_start+HID_DIM])
    elif REGION == "snr":
        flow_field.fit_pca(sampled_acts[:, snr_start:snr_start+HID_DIM])
    elif REGION == "str2thal":
        flow_field.fit_pca(sampled_acts[:, str_start:alm_start])

    grid = flow_field.inverse_pca(data_coords)
    grid = torch.tensor(grid, device="cuda").clone().detach()

    if PERTURBATION == True:
        len_seq_act = len_seq[CONDITION]
    else:
        len_seq_act = len_seq[CONDITION]
    
    # Get perturbed trajectory
    if PERTURBATION  == True:

        perturbed_acts = []
        hn = torch.zeros(size=(1, 1, total_num_units)).cuda()
        xn = hn

        for t in range(len_seq_act):

            if t < ITI_steps:
                inp = x_data[CONDITION:CONDITION+1, 0:1, :]
            else:
                inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]

            with torch.no_grad():        

                if PERTURBED_REGION == "alm" and t > start_silence and t < end_silence:
                    inhib_stim = (-10 * alm_mask).unsqueeze(0).unsqueeze(0)
                    inp = 0*x_data[CONDITION:CONDITION+1, 0:1, :]
                elif PERTURBED_REGION == "str" and t > start_silence and t < end_silence:
                    inhib_stim = (-1 * str_mask).unsqueeze(0).unsqueeze(0)
                    #inhib_stim = 1 * str_mask
                else:
                    inhib_stim = torch.zeros(size=(1, 1, hn.shape[-1]), device="cuda")
                        
                hn, _, xn, _ = rnn(inp, hn, xn, inhib_stim, noise=False)
                perturbed_acts.append(hn)


    # initialize activity dict
    next_acts = {}
    for i in range(500, len_seq_act, TIME_SKIPS):
        next_acts[i] = []

    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()
    xn = hn

    if PERTURBATION == True:
        hidden_input = torch.concatenate(perturbed_acts, dim=1).clone().detach().cuda()
    else:
        hidden_input = torch.tensor(sampled_acts).clone().detach().unsqueeze(0).cuda()

    inhib_stim = torch.zeros(size=(1, 1, hn.shape[-1]), device="cuda")

    # Go through activities and generate h_t+1
    for t in range(500, len_seq_act, TIME_SKIPS):

        print(f"input number: {t}")

        if t < ITI_steps:
            inp = x_data[CONDITION:CONDITION+1, 0:1, :]
        else:
            inp = x_data[CONDITION:CONDITION+1, ITI_steps+1:ITI_steps+2, :]
        
        if PERTURBATION == True and REGION == "alm" and t > start_silence and t < end_silence:
            inp = 0 * x_data[CONDITION:CONDITION+1, 0:1, :]
        
        for h_0 in grid:

            with torch.no_grad():

                h_0 = h_0.unsqueeze(0).cuda()

                if REGION == "str":
                    h_0 = torch.cat([h_0, hidden_input[0, t:t+1, str_start+HID_DIM:]], dim=1).unsqueeze(0)
                elif REGION == "str2thal":
                    h_0 = torch.cat([h_0, hidden_input[0, t:t+1, alm_start:]], dim=1).unsqueeze(0)
                elif REGION == "alm":
                    h_0 = torch.cat([hidden_input[0, t:t+1, :alm_start], h_0], dim=1).unsqueeze(0)
                elif REGION == "snr":
                    h_0 = torch.cat([hidden_input[0, t:t+1, :snr_start], h_0, hidden_input[0, t:t+1, snr_start+HID_DIM:]], dim=1).unsqueeze(0)

                _, act, _, _ = rnn(inp, h_0, xn, inhib_stim, noise=False)

                if REGION == "str": 
                    next_acts[t].append(act[0, 0, :str_start+HID_DIM].detach().cpu().numpy())
                elif REGION == "str2thal": 
                    next_acts[t].append(act[0, 0, :alm_start].detach().cpu().numpy())
                elif REGION == "alm":
                    next_acts[t].append(act[0, 0, alm_start:].detach().cpu().numpy())
                elif REGION == "snr":
                    next_acts[t].append(act[0, 0, snr_start:snr_start+HID_DIM].detach().cpu().numpy())

    # Reshape data back to grid
    data_coords = data_coords.numpy()
    data_coords = np.reshape(data_coords, (NUM_POINTS, NUM_POINTS, data_coords.shape[-1]))

    for i in range(500, len_seq_act, TIME_SKIPS):

        next_acts[i] = flow_field.transform_pca(np.array(next_acts[i]))
        next_acts[i] = np.reshape(next_acts[i], (NUM_POINTS, NUM_POINTS, next_acts[i].shape[-1]))
    
    if REGION == "str":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, :str_start+HID_DIM])
    elif REGION == "alm":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, alm_start:])
    elif REGION == "snr":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, snr_start:snr_start+HID_DIM])
    elif REGION == "str2thal":
        sampled_acts_region = flow_field.transform_pca(sampled_acts[:, :alm_start])
    
    x_vels = {}
    y_vels = {}

    for i in range(500, len_seq_act, TIME_SKIPS):

        x_vels[i] = next_acts[i][:, :, 0] - data_coords[:, :, 0]
        y_vels[i] = next_acts[i][:, :, 1] - data_coords[:, :, 1]
    
    speeds = {}
    
    for i in range(500, len_seq_act, TIME_SKIPS):
        speed = np.sqrt(x_vels[i]**2 + y_vels[i]**2)
        c = speed / speed.max()
        speeds[i] = c

    for i in range(500, len_seq_act, TIME_SKIPS):

        plt.scatter(sampled_acts_region[i, 0], sampled_acts_region[i, 1], c="red", s=250)
        plt.streamplot(x, y, x_vels[i], y_vels[i], color=speeds[i], cmap="plasma", linewidth=3, arrowsize=2)
        plt.yticks([])
        plt.xticks([])
        plt.savefig(SAVE_NAME + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.png")
        plt.savefig(SAVE_NAME_EPS + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.eps")
        plt.close()
    
if __name__ == "__main__":
    main()