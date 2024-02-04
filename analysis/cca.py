from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle

rnn_fr = np.load("lick_ramp8000_fr.npy")
 
#First filter the agent's activity with 20ms gaussian as done with experimental activity during preprocessing
A_agent = gaussian_filter1d(rnn_fr.T, 20).T

#Reduce the activity using PCA to the first 10 components
PC_agent = PCA(n_components= 10)
PC_exp = PCA(n_components= 10)

A_exp = PC_exp.fit_transform(A_exp)
A_agent = PC_agent.fit_transform(A_agent)

#Do the CCA
cca = CCA(n_components=10)
U_c, V_c = cca.fit_transform(A_exp, A_agent)

result = np.corrcoef(U_c[:,9], V_c[:,9])
print(result)
U_prime = cca.inverse_transform(V_c)