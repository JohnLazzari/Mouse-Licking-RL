from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
import scipy.io as sio
from sklearn.metrics import r2_score

RNN_PATH = "results/test_activity/lick_ramp8000_fr.npy"
DATA_PATH = "data/firing_rates/striatum_fr_population_cond1.mat"

def main():
    
    rnn_fr = np.load(RNN_PATH)
    A_exp = sio.loadmat(DATA_PATH)['fr_population'][:rnn_fr.shape[0], :]
    A_agent = gaussian_filter1d(rnn_fr, 10, axis=0)

    ridge = Ridge(alpha=1e-1)
    print(r2_score(A_exp[:, 0], pred[:, 0]))
    ridge.fit(A_agent, A_exp)

    print(ridge.score(A_agent, A_exp))

    # Make Prediction
    pred = ridge.predict(A_agent)

    for i in range(A_exp.shape[1]):
        print(r2_score(A_exp[:, i], pred[:, i]))

if __name__ == "__main__":
    main()