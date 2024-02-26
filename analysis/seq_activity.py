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
        x_i = np.random.exponential(scale=lambda_max)
        x_is.append(x_i)
        u = 0
        for k in range(i):
            u += x_is[k]
        p = fr[i] / lambda_max
        rand = np.random.uniform(0, 1)
        if rand < p:
            spike_times.append(u)

    print(spike_times)
    return spike_times

rnn_fr = np.load("results/rnn_control_regression_act.npy")

cond_1_act = rnn_fr[0, :110, :]
cond_2_act = rnn_fr[1, :119, :]
cond_3_act = rnn_fr[2, :128, :]

cond_1_act_sorted, ordering = sort_by_peaks(cond_1_act)

# Currently N x T
spike_times = []
neuron_index = []
for i, neuron in enumerate(cond_1_act_sorted):
    spike_times.append(thinning(neuron))
    neuron_index.append([i]*len(spike_times[-1]))
    assert len(spike_times[-1]) == len(neuron_index[-1])

spike_times = list(chain.from_iterable(spike_times))
neuron_index = list(chain.from_iterable(neuron_index))

plt.scatter(spike_times, neuron_index)
plt.show()
#plt.plot(cond_1_act_sorted.T)
#plt.show()

rnn_fr = np.load("results/lick_ramp10000_fr.npy")
rnn_fr = gaussian_filter1d(rnn_fr, 10, axis=0)
ramp_sorted, ordering = sort_by_peaks(rnn_fr)
#plt.plot(ramp_sorted.T[:, :30])
#plt.show()