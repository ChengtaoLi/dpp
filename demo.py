import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.distance import pdist, squareform

import sampler.dpp as dpp
import sampler.utils as utils

# currently only support cpu mode
flag_gpu = False

# Construct kernel matrix
Ngrid = 100
X = np.mgrid[-2:2:4./Ngrid, -2:2:4./Ngrid].reshape(2,Ngrid**2).transpose()
pairwise_dists = squareform(pdist(X, 'euclidean'))
L = np.exp(-pairwise_dists ** 2 / 0.5 ** 2)

# Get eigendecomposition of kernel matrix
D, V = utils.get_eig(L, flag_gpu=flag_gpu)

# Samples and plot from unif and standard DPPs
dpp_smpl  = dpp.sample_dpp(D, V, flag_gpu=flag_gpu)
unif_smpl = np.random.permutation(len(X))[:len(dpp_smpl)]

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(X[unif_smpl, 0], X[unif_smpl, 1],'r.',)
plt.title('Unif')

plt.subplot(1,2,2)
plt.plot(X[dpp_smpl, 0], X[dpp_smpl, 1],'b.',)
plt.title('DPP')

plt.savefig('fig/unif-dpp', bbox_inches='tight')


# Samples and plot from unif and k-DPPs
k = 100
E = utils.get_sympoly(D, k, flag_gpu=flag_gpu)

# Samples and plot from unif and standard DPPs
unif_smpl = np.random.permutation(len(X))[:k]
dpp_smpl  = dpp.sample_dpp(D, V, E=E, k=k, flag_gpu=flag_gpu)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(X[unif_smpl, 0], X[unif_smpl, 1],'r.',)
plt.title('Unif')

plt.subplot(1,2,2)
plt.plot(X[dpp_smpl, 0], X[dpp_smpl, 1],'b.',)
plt.title('k-DPP')

plt.savefig('fig/unif-kdpp', bbox_inches='tight')


