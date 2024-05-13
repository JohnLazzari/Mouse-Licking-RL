from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
import scipy.io as sio
from sklearn.metrics import r2_score

plt.rcParams.update({'font.size': 16})

RNN_PATH_CONSTRAINED = "results/test_activity/lick_attractor_lowd_trajectory_constrained4_act.npy"
RNN_PATH_UNCONSTRAINED = "results/test_activity/lick_attractor_lowd_act.npy"
DATA_PATH = "data/firing_rates/striatum_fr_population_cond1.mat"

def main():
    
    rnn_fr_constrained = np.load(RNN_PATH_CONSTRAINED)
    rnn_fr_unconstrained = np.load(RNN_PATH_UNCONSTRAINED)

    A_exp_constrained = sio.loadmat(DATA_PATH)['fr_population'][100:100+rnn_fr_constrained.shape[0], :]
    A_exp_unconstrained = sio.loadmat(DATA_PATH)['fr_population'][100:100+rnn_fr_unconstrained.shape[0], :]

    A_agent_constrained = gaussian_filter1d(rnn_fr_constrained, 10, axis=0)
    A_agent_unconstrained = gaussian_filter1d(rnn_fr_unconstrained, 10, axis=0)

    ridge_constrained = Ridge(alpha=1e-1)
    ridge_unconstrained = Ridge(alpha=1e-1)

    ridge_constrained.fit(A_agent_constrained, A_exp_constrained)
    ridge_unconstrained.fit(A_agent_unconstrained, A_exp_unconstrained)

    # Make Prediction
    pred_constrained = ridge_constrained.predict(A_agent_constrained)
    pred_unconstrained = ridge_unconstrained.predict(A_agent_unconstrained)

    r2_constrained = []
    for i in range(A_exp_constrained.shape[1]):
        r2_constrained.append(r2_score(A_exp_constrained[:, i], pred_constrained[:, i]))

    r2_unconstrained = []
    for i in range(A_exp_unconstrained.shape[1]):
        r2_unconstrained.append(r2_score(A_exp_unconstrained[:, i], pred_unconstrained[:, i]))

    plt.scatter(r2_unconstrained, r2_constrained, s=10)
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), '--r')
    plt.xlabel("Unconstrained R^2")
    plt.ylabel("Constrained R^2")
    plt.show()

    for k in range(2):
        if k==0:
            plt.plot(A_exp_unconstrained[:,k]/np.max(A_exp_unconstrained[:,k]) + k*3, linewidth=1.5, c = 'k')
            plt.plot(pred_unconstrained[:,k]/np.max(A_exp_unconstrained[:,k]) + k*3, linewidth= 1.5, color="#659EFF", label='Network Reconstruction')
        else:
            plt.plot(A_exp_unconstrained[:, k]/np.max(A_exp_unconstrained[:,k]) + k * 3, linewidth=1.5, c='k')
            plt.plot(pred_unconstrained[:, k]/np.max(A_exp_unconstrained[:,k]) + k * 3, linewidth=1.5, color="#659EFF")

    for k in range(2, 10, 1):
        if k==0:
            plt.plot(A_exp_unconstrained[:,523+k]/np.max(A_exp_unconstrained[:,523+k]) + k*3, linewidth=1.5, c = 'k')
            plt.plot(pred_unconstrained[:,523+k]/np.max(A_exp_unconstrained[:,523+k]) + k*3, linewidth= 1.5, color="#659EFF")
        else:
            plt.plot(A_exp_unconstrained[:, 523 + k]/np.max(A_exp_unconstrained[:,523+k]) + k * 3, linewidth=1.5, c='k')
            plt.plot(pred_unconstrained[:, 523 + k]/np.max(A_exp_unconstrained[:,523+k]) + k * 3, linewidth=1.5, color="#659EFF")
    
    plt.ylabel('Neuron Firing Rate', size=14)
    plt.yticks([])
    plt.xticks([])
    plt.title(f"Linear Regression Reconstruction")
    plt.show()

    plt.plot(A_exp_unconstrained[:, 100:200], '--k')
    plt.plot(pred_unconstrained[:, 100:200], '-', color="#659EFF")
    plt.show()

if __name__ == "__main__":
    main()