from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
import scipy.io as sio
import scipy.signal as sig
from itertools import chain

def sort_by_peaks(fr):
    
    # fr: T x N
    peaks_list = []
    included_neurons = np.zeros(shape=(1, fr.shape[0]))

    # Takes in all of the neurons
    for neuron in fr.T:

        indices, properties = sig.find_peaks(neuron)
        if len(indices) != 0:
            max_index = np.argmax(neuron[indices])
            peaks_list.append(indices[max_index])
            included_neurons = np.concatenate((included_neurons, np.expand_dims(neuron, axis=0)), axis=0)
    included_neurons = included_neurons[1:]

    sorted_peaks = np.argsort(np.array(peaks_list))
    sorted_neurons = included_neurons[sorted_peaks]

    return sorted_neurons, sorted_peaks

def thinning(fr):
    
    # fr: N X 1

    # Takes in a single neuron
    lambda_max = np.max(fr)
    spike_times = []
    x_is = []
    for i in range(fr.shape[0]):
        x_i = np.random.exponential(scale=lambda_max) # keep this as lambda not 1/lambda
        x_is.append(x_i)
        u = 0
        for k in range(i):
            u += x_is[k]
        p = fr[i] / lambda_max
        rand = np.random.uniform(0, 1)
        if rand < p:
            spike_times.append(u)

    return spike_times

rnn_fr = np.load("results/test_activity/lick_attractor_act.npy")
ramp_sorted, ordering = sort_by_peaks(np.abs(rnn_fr))

plt.plot(ramp_sorted.T)
plt.show()

plt.imshow(np.abs(ramp_sorted))
plt.show()