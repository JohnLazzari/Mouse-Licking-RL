from sklearn.cross_decomposition import CCA
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import pickle
import scipy.io as sio

rnn_fr = np.load("results/lick_ramp30000_fr.npy")
A_exp = sio.loadmat("data/striatum_fr_population_1.1s.mat")['fr_population'][:rnn_fr.shape[0], :]
 
#First filter the agent's activity with 20ms gaussian as done with experimental activity during preprocessing
A_agent = gaussian_filter1d(rnn_fr, 10, axis=0)

#Reduce the activity using PCA to the first 10 components
PC_agent = PCA(n_components=10)
PC_exp = PCA(n_components=10)

A_exp = PC_exp.fit_transform(A_exp)
A_agent = PC_agent.fit_transform(A_agent)
plt.plot(A_agent[:, 0])
plt.show()

#Do the CCA
cca = CCA(n_components=10)
U_c, V_c = cca.fit_transform(A_exp, A_agent)

result = np.corrcoef(U_c[:,0], V_c[:,0])
sum = 0
for k in range(10):
    sum = sum + np.corrcoef(U_c[:, k], V_c[:, k])[0, 1]
average = sum / 10
print(average)
U_prime = cca.inverse_transform(V_c)

for k in range(10):
    if k==0:
        plt.plot(A_exp[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth=1.5, c = 'k')
        plt.plot(U_prime[:,9-k]/np.max(A_exp[:,9-k]) + k*4, linewidth= 1.5, c=(50/255, 205/255, 50/255), label= 'Network Reconstruction')
    else:
        plt.plot(A_exp[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c='k')
        plt.plot(U_prime[:, 9 - k]/np.max(A_exp[:,9-k]) + k * 4, linewidth=1.5, c=(50 / 255, 205 / 255, 50 / 255))

plt.ylabel('Reconstructed M1 Population Activity', size=14)
#plt.xticks([0, 226], ['0', '0.5'], size= 14)
plt.yticks([])
# plt.legend()
# plt.savefig('C:/Users/malma/Dropbox/NatureFigs2/Fig2/CCA_619.svg', format='svg', dpi=300, transparent= True)
plt.title(f"Inverse CCA Train Condition {1}")
plt.show()

#Now plot the PCs on the same plot here
ax = plt.figure(figsize= (6,6), dpi=100).add_subplot(projection='3d')
ax.plot(A_exp[:,0], A_exp[:, 1], A_exp[:, 2], c = 'k')
ax.plot(U_prime[:,0], U_prime[:, 1], U_prime[:, 2], c=(50/255, 205/255, 50/255))

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

sum = 0
for k in range(10):
    sum = sum + np.corrcoef(A_exp[:, k], U_prime[:, k])[0, 1]
average = sum / 10

print(f"Correlation for Train Condition {1}", average)