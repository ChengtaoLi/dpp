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
Ngrid = 5
X = np.mgrid[-2:2:4./Ngrid, -2:2:4./Ngrid].reshape(2,Ngrid**2).transpose()
pairwise_dists = squareform(pdist(X, 'euclidean'))
L = np.exp(-pairwise_dists ** 2 / 4 ** 2)

print(X)
print(utils.kpp(X, 5, flag_kernel=False))

a = np.array([1,2,3,4,5])
b = np.array([2,4])
print(np.setdiff1d(a, b))
mcdpp.sample(L, 20, k=5)
