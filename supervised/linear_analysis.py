import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from models import CBGTCL
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.decomposition import PCA
import scipy
import scipy.io as sio
from utils import load_variables_from_file

plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
font = {'size' : 16}
plt.rc('font', **font)
plt.rcParams['axes.linewidth'] = 2 # set the value globally
plt.rcParams['figure.figsize'] = [10, 8]

MODEL_NAMES = ["22_11_2024_15_13_46"]
SPECIFICATIONS_PATH = "checkpoints/model_specifications/"
CHECK_PATH = "checkpoints/"
SAVE_NAME_PATH = "results/linear_analysis/"
DATA_SPLIT = "training"
START_SILENCE = 160
END_SILENCE = 220
STIM_STRENGTH = -5
EXTRA_STEPS = 100
REGIONS_CELL_TYPES = [("alm", "exc")]

def plot_eigenvalues(mrnn):

    mrnn_weight = mrnn.apply_dales_law()
    weight_subset = mrnn_weight[:mrnn.total_num_units - mrnn.region_dict["alm"].num_units, 
                                :mrnn.total_num_units - mrnn.region_dict["alm"].num_units]
    weight_subset = weight_subset.detach().cpu().numpy()

    eigenvalues, eigenvectors = np.linalg.eig(weight_subset.T)
    
    reals = []
    for eigenvalue in eigenvalues:
        reals.append(eigenvalue.real)

    ims = []
    for eigenvalue in eigenvalues:
        ims.append(eigenvalue.imag)

    t = np.linspace(0, 2*np.pi)
    x_c = np.sin(t)
    y_c = np.cos(t)

    plt.scatter(reals, ims, s=150, alpha=0.5, c="red")
    plt.plot(x_c, y_c, c="black", linewidth=2)
    plt.axhline(y=0, c="black")
    plt.axvline(x=0, c="black")
    plt.ylim(-1.5, 1.5)
    plt.xlim(-1.5, 1.5)
    plt.xlabel("Real Axis")
    plt.ylabel("Im Axis")
    plt.show()

def main():
    
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

        checkpoint = torch.load(CHECK_PATH + model + ".pth")
        rnn = CBGTCL(config_name, inp_dim, out_dim, out_type, constrained=constrained).cuda()
        rnn.load_state_dict(checkpoint)

        plot_eigenvalues(rnn.mrnn)

if __name__ == "__main__":
    main()