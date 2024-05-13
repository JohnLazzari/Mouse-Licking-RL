from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
import scipy.io as sio

RNN_PATH = "results/test_activity/lick_attractor_lowd_act.npy"
DATA_PATH = "data/firing_rates/striatum_fr_population_cond1.mat"
plt.rcParams.update({'font.size': 16})

def main():

    rnn_fr = np.load(RNN_PATH)
    A_exp = sio.loadmat(DATA_PATH)['fr_population'][100:100+rnn_fr.shape[0], :]

    #First filter the agent's activity with 20ms gaussian as done with experimental activity during preprocessing
    A_agent = gaussian_filter1d(rnn_fr, 10, axis=0)

    plt.plot(A_exp)
    plt.show()

    plt.plot(A_agent)
    plt.show()

    #Reduce the activity using PCA to the first 10 components
    PC_agent = PCA(n_components=10)
    PC_exp = PCA(n_components=10)

    A_exp = PC_exp.fit_transform(A_exp)
    A_agent = PC_agent.fit_transform(A_agent)

    t = np.linspace(-1, rnn_fr.shape[0] * 0.01, rnn_fr.shape[0])

    plt.plot(t, A_agent[:, 0])
    plt.ylabel("FR")
    plt.xlabel("Time")
    plt.title("PC 1")
    plt.show()

    plt.plot(t, A_agent[:, 1])
    plt.ylabel("FR")
    plt.xlabel("Time")
    plt.title("PC 2")
    plt.show()

    plt.plot(t, A_agent[:, 2])
    plt.ylabel("FR")
    plt.xlabel("Time")
    plt.title("PC 3")
    plt.show()

    #Do the CCA
    cca = CCA(n_components=10)
    U_c, V_c = cca.fit_transform(A_exp, A_agent)
    U_prime = cca.inverse_transform(V_c)

    sum = 0
    for k in range(10):
        sum = sum + np.corrcoef(A_exp[:, k], U_prime[:, k])[0, 1]
    average = sum / 10

    for k in range(10):
        if k==0:
            plt.plot(A_exp[:,9-k]/np.max(A_exp[:,9-k]) + k*3, linewidth=1.5, c = 'k')
            plt.plot(U_prime[:,9-k]/np.max(A_exp[:,9-k]) + k*3, linewidth= 1.5, color="#659EFF", label='Network Reconstruction')
        else:
            plt.plot(A_exp[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 3, linewidth=1.5, c='k')
            plt.plot(U_prime[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 3, linewidth=1.5, color="#659EFF")

    plt.ylabel('Reconstructed Striatal Population Activity', size=14)
    plt.yticks([])
    plt.xticks([])
    plt.xlabel("Time")
    plt.title(f"Inverse CCA")
    plt.show()

    #Now plot the PCs on the same plot here
    ax = plt.figure(figsize= (6,6), dpi=100).add_subplot(projection='3d')
    ax.plot(A_exp[:,0], A_exp[:, 1], A_exp[:, 2], c = 'k')
    ax.plot(U_prime[:,0], U_prime[:, 1], U_prime[:, 2], color="#659EFF")

    # Hide grid lines
    ax.grid(False)
    plt.grid(b=None)

    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    plt.axis('off')
    plt.title(f"PC plot Train Condition {1}")

    plt.show()

    print(f"Correlation for Train Condition {1}", average)

if __name__ == "__main__":
    main()