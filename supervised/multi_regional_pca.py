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
MODEL_TYPE = "d1d2" # constraint, no_constraint, no_constraint_thal
REGION = "alm" # str, alm, or str2thal
CONSTRAINED = False
CHECK_PATH = f"checkpoints/rnn_goal_data_multiregional_bigger_long_conds_localcircuit_ramping_{MODEL_TYPE}_unconstrained.pth"

class FlowFields():
    def __init__(self, dimensions=2):
        
        self.dimensions = dimensions
        self.x_pca = PCA(n_components=dimensions)
    
    def generate_grid(self, num_points, lower_bound=-10, upper_bound=15):

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
    if MODEL_TYPE == "d1d2":
        rnn = RNN_MultiRegional_D1D2(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
    if MODEL_TYPE == "d1":
        rnn = RNN_MultiRegional_D1(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()
    if MODEL_TYPE == "stralm":
        rnn = RNN_MultiRegional_STRALM(INP_DIM, HID_DIM, OUT_DIM, noise_level=0.01, constrained=CONSTRAINED).cuda()

    rnn.load_state_dict(checkpoint)

    if MODEL_TYPE == "d1d2":

        total_num_units = HID_DIM * 6
        str_start = 0
        snr_start = HID_DIM*3
        thal_start = HID_DIM*4
        alm_start = HID_DIM*5

    elif MODEL_TYPE == "d1":

        total_num_units = HID_DIM * 3
        str_start = 0
        thal_start = HID_DIM
        alm_start = HID_DIM*2

    elif MODEL_TYPE == "stralm":

        total_num_units = HID_DIM * 2
        str_start = 0
        alm_start = HID_DIM

    # Gather data
    flow_field = FlowFields()

    # Get input and output data
    x_data, len_seq = gather_inp_data(dt=DT, hid_dim=HID_DIM)
    x_data = x_data.cuda()

    x_data[0, len_seq[0]:, :] = x_data[0, 1001:1002, :]
    x_data[1, len_seq[1]:, :] = x_data[1, 1001:1002, :]

    # Sample many hidden states to get pcs for dimensionality reduction
    hn = torch.zeros(size=(1, 1, total_num_units)).cuda()
    xn = hn

    inhib_stim = torch.zeros(size=(1, x_data.shape[1], hn.shape[-1]), device="cuda")

    # Get original trajectory
    with torch.no_grad():
        _, orig_acts, _, _ = rnn(x_data, hn, xn, inhib_stim, noise=False)

    orig_acts = orig_acts[:, 500:, :]
    acts = orig_acts
    acts = torch.reshape(acts, shape=(acts.shape[0] * acts.shape[1], acts.shape[2])) 
    acts = acts.detach().cpu().numpy()

    if REGION == "str":
        flow_field.fit_pca(acts[:, :str_start+HID_DIM])
        reduced_trajectory = flow_field.transform_pca(acts[:, :str_start+HID_DIM])
        reduced_trajectory = reduced_trajectory.reshape((orig_acts.shape[0], orig_acts.shape[1], 2))
    elif REGION == "alm":
        flow_field.fit_pca(acts[:, alm_start:])
        reduced_trajectory = flow_field.transform_pca(acts[:, alm_start:])
        reduced_trajectory = reduced_trajectory.reshape((orig_acts.shape[0], orig_acts.shape[1], 2))
    elif REGION == "str2thal":
        flow_field.fit_pca(acts[:, :alm_start])
        reduced_trajectory = flow_field.transform_pca(acts[:, :alm_start])
        reduced_trajectory = reduced_trajectory.reshape((orig_acts.shape[0], orig_acts.shape[1], 2))
    
    print(flow_field.x_pca.explained_variance_ratio_)

    plt.plot(reduced_trajectory[0, :, 0], reduced_trajectory[0, :, 1], linewidth=4, color="red")
    plt.plot(reduced_trajectory[1, :, 0], reduced_trajectory[1, :, 1], linewidth=4, color="blue")
    plt.plot(reduced_trajectory[2, :, 0], reduced_trajectory[2, :, 1], linewidth=4, color="green")

    plt.tick_params(left=False, bottom=False) 
    plt.xticks([])
    plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()