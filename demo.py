import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from scipy.spatial.distance import pdist, squareform

import sampler.dpp as dpp
import sampler.mcdpp as mcdpp
import sampler.utils as utils

# currently only support cpu mode
flag_gpu = False

# Construct kernel matrix
Ngrid = 50
X = np.mgrid[-1:1:2./Ngrid, -1:1:2./Ngrid].reshape(2,Ngrid**2).transpose()
pairwise_dists = squareform(pdist(X, 'euclidean'))
L = np.exp(-pairwise_dists ** 2 / 0.5 ** 2)

# Get eigendecomposition of kernel matrix
D, V = utils.get_eig(L, flag_gpu=flag_gpu)

# Samples and plot from unif and standard DPPs
print('DPP-Eigendecomp')
dpp_smpl  = dpp.sample(D, V, flag_gpu=flag_gpu)
mc_init = utils.kpp(L, len(X), flag_kernel=True)

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
print('kDPP-Eigendecomp')
dpp_smpl  = dpp.sample(D, V, E=E, k=k, flag_gpu=flag_gpu)

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(X[unif_smpl, 0], X[unif_smpl, 1],'r.',)
plt.title('Unif')

plt.subplot(1,2,2)
plt.plot(X[dpp_smpl, 0], X[dpp_smpl, 1],'b.',)
plt.title('kDPP')

plt.savefig('fig/unif-kdpp', bbox_inches='tight')


