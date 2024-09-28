import numpy as np
import matplotlib.pyplot as plt
import torch
from models import RNN_MultiRegional_D1D2, RNN_MultiRegional_STRALM
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy.io as sio
from utils import gather_inp_data, get_region_borders
from FixedPointFinderTorch import FixedPointFinderTorch
import torch.nn.functional as F

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False

HID_DIM = 256 # Hid dim of each region
OUT_DIM = 1451
INP_DIM = int(HID_DIM*0.1)
DT = 1e-2
CONDITION = 1
NUM_POINTS = 100
MODEL_TYPE = "d1d2" # d1d2, d2, stralm
REGION = "str2thal" # str, alm, str2thal, or snr
TIME_SKIPS = 10
PERTURBATION = False
PERTURBED_REGION = "alm" # str or alm
ITI_STEPS = 100
ALM_PERTURBATION_TYPE = "none" # onlyramp, onlyITI, or none (none is both ITI and ramp are silenced)
CONSTRAINED = True
NUM_POINTS_OPTIMIZATION = 250
START_SILENCE = 160
END_SILENCE = 220
LOWER_BOUND_X = -15
UPPER_BOUND_X = 15
LOWER_BOUND_Y = -15
UPPER_BOUND_Y = 15
TRIAL_EPOCH = "full"
INP_PATH = "data/firing_rates/ITIProj_trialPlotAll1.mat"
CHECK_PATH = f"checkpoints/{MODEL_TYPE}_datadriven_itiinp_full_256n_almnoise.05_itinoise.05_15000iters_newloss.pth"
SAVE_NAME = f"results/flow_fields/multi_regional_{MODEL_TYPE}/multi_regional_{MODEL_TYPE}_flow"
SAVE_NAME_EPS = f"results/flow_fields/multi_regional_{MODEL_TYPE}_eps/multi_regional_{MODEL_TYPE}_flow"

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


def main():
    
    fsi_size = int(HID_DIM * 0.3)
    checkpoint = torch.load(CHECK_PATH)
    
    # Create RNN
    rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level_act=0.0, noise_level_inp=0.0, constrained=CONSTRAINED).cuda()
    rnn.load_state_dict(checkpoint)

    # Region masks
    alm_mask = rnn.alm_ramp_mask
    str_mask = rnn.str_d1_mask
    total_num_units = rnn.total_num_units

    # Get region borders
    region_start, region_end = get_region_borders(
        MODEL_TYPE, 
        REGION, 
        HID_DIM, 
        INP_DIM
    )

    # Gather data
    flow_field = FlowFields()

    # Get input and output data
    iti_inp, cue_inp, len_seq = gather_inp_data(DT, HID_DIM, INP_PATH, TRIAL_EPOCH)
    iti_inp, cue_inp = iti_inp.cuda(), cue_inp.cuda()

    # Generate a grid which represents combinations of values of the first two PCs of the network
    x, y, data_coords = flow_field.generate_grid(
        num_points=NUM_POINTS,
        lower_bound_x=LOWER_BOUND_X,
        upper_bound_x=UPPER_BOUND_X,
        lower_bound_y=LOWER_BOUND_Y,
        upper_bound_y=UPPER_BOUND_Y
    )

    # Initialize hidden states
    hn = torch.zeros(size=(1, 4, total_num_units)).cuda()
    xn = torch.zeros(size=(1, 4, total_num_units)).cuda()
    inhib_stim = torch.zeros(size=(1, max(len_seq), hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        act, _ = rnn(iti_inp, cue_inp, hn, xn, inhib_stim, noise=False)

    # Reshape before PCA
    # For now, using all of the conditions for PCA
    sampled_acts = torch.reshape(act, shape=(act.shape[0] * act.shape[1], act.shape[2])) 
    sampled_acts = sampled_acts.detach().cpu().numpy()

    # Do PCA on the specified region
    flow_field.fit_pca(sampled_acts[:, region_start:region_end])

    # Reshape sampled acts back to normal
    sampled_acts = torch.reshape(act, shape=(act.shape[0], act.shape[1], act.shape[2])) 

    # Inverse PCA to input grid into network
    grid = flow_field.inverse_pca(data_coords)
    grid = grid.clone().detach().cuda()

    # initialize activity dict
    next_acts = {}

    # Gather input
    inhib_stim = torch.zeros(size=(grid.shape[0], 1, total_num_units), device="cuda")

    # Go through activities and generate h_t+1
    for t in range(50, max(len_seq), TIME_SKIPS):

        print(f"input number: {t}")

        with torch.no_grad():

            if REGION == "str":

                h_0 = F.relu(torch.cat([grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0))
                x_0 = torch.cat([grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0)

            elif REGION == "str2thal":

                h_0 = F.relu(torch.cat([grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0))
                x_0 = torch.cat([grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0)

            elif REGION == "alm":
                
                h_0 = F.relu(torch.cat([sampled_acts[CONDITION, t:t+1, :region_start].repeat(grid.shape[0], 1), grid], dim=1).unsqueeze(0))
                x_0 = torch.cat([sampled_acts[CONDITION, t:t+1, :region_start].repeat(grid.shape[0], 1), grid], dim=1).unsqueeze(0)

            elif REGION == "snr":

                h_0 = F.relu(torch.cat([sampled_acts[CONDITION, t:t+1, :region_start].repeat(grid.shape[0], 1), grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0))
                x_0 = torch.cat([sampled_acts[CONDITION, t:t+1, :region_start].repeat(grid.shape[0], 1), grid, sampled_acts[CONDITION, t:t+1, region_end:].repeat(grid.shape[0], 1)], dim=1).unsqueeze(0)

            act, _ = rnn(iti_inp[CONDITION:CONDITION+1, t:t+1, :], cue_inp[CONDITION:CONDITION+1, t:t+1, :], h_0, x_0, inhib_stim, noise=False)

            next_acts[t] = act[:, 0, region_start:region_end].detach().cpu().numpy()
        
        act_max, act_min = torch.max(h_0), torch.min(h_0)
        print("Max: ", act_max)
        print("Min: ", act_min)

    # Reshape data back to grid
    data_coords = data_coords.numpy()
    data_coords = np.reshape(data_coords, (NUM_POINTS, NUM_POINTS, data_coords.shape[-1]))

    sampled_acts_region = sampled_acts[CONDITION, :, region_start:region_end].detach().cpu().numpy()
    sampled_acts_region = flow_field.transform_pca(sampled_acts_region)

    for t in range(50, max(len_seq), TIME_SKIPS):

        next_acts[t] = flow_field.transform_pca(np.array(next_acts[t]))
        next_acts[t] = np.reshape(next_acts[t], (NUM_POINTS, NUM_POINTS, next_acts[t].shape[-1]))

    x_vels = {}
    y_vels = {}

    for i in range(50, max(len_seq), TIME_SKIPS):

        x_vels[i] = next_acts[i][:, :, 0] - data_coords[:, :, 0]
        y_vels[i] = next_acts[i][:, :, 1] - data_coords[:, :, 1]
    
    speeds = {}
    
    for i in range(50, max(len_seq), TIME_SKIPS):
        speed = np.sqrt(x_vels[i]**2 + y_vels[i]**2)
        c = speed / speed.max()
        speeds[i] = c

    for i in range(50, max(len_seq), TIME_SKIPS):
        
        '''
        print("timestep: ", i)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        surf = ax.plot_surface(data_coords[:, :, 0], data_coords[:, :, 1], speeds[i][:, :], cmap=plt.cm.coolwarm,
                       linewidth=0, antialiased=False)
        plt.tight_layout()
        plt.show()
        plt.close()
        '''
        
        plt.plot(sampled_acts_region[:i, 0], sampled_acts_region[:i, 1], c="black", linewidth=4, zorder=5)
        plt.streamplot(x, y, x_vels[i], y_vels[i], color=speeds[i], cmap="plasma", linewidth=3, arrowsize=2, zorder=0)
        plt.yticks([])
        plt.xticks([])

        plt.savefig(SAVE_NAME + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.png")
        plt.savefig(SAVE_NAME_EPS + f"_perturbation_{PERTURBATION}_region_{REGION}_cond{CONDITION}_inp{i}.eps")

        plt.close()
    
if __name__ == "__main__":
    main()