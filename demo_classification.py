import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import inv, norm, sqrtm

import sampler.dpp as dpp
import sampler.utils as utils

import helper.lr as lr

# currently only support cpu mode
flag_gpu = False
nTrn = 4000
nTst = 1000
trnX, tstX, trnY, tstY = utils.load_mnist(ntrain=nTrn, ntest=nTst)

print(trnX.shape)
print(trnY.shape)
print(tstX.shape)
pairwise_dists = squareform(pdist(np.concatenate((trnX, tstX)), 'euclidean'))
L = np.exp(-pairwise_dists ** 2 / 100 ** 2)
trnL = L[:nTrn, :nTrn]

k_group = [20,30,50,70,100]
error_unif = np.zeros(len(k_group))
error_dpp = np.zeros(len(k_group))

for k_idx in xrange(len(k_group)):
	k = k_group[k_idx]
	# Uniform sampling
	unif_smpl = np.random.permutation(nTrn)[:k]

	C = L[np.ix_(range(nTrn+nTst), unif_smpl)]
	W = C[np.ix_(unif_smpl, range(k))]
	X_prime = C.dot(inv(np.real(sqrtm(W))))
	trnX_prime = X_prime[:nTrn]
	tstX_prime = X_prime[nTrn:]
	
	error_unif[k_idx] = lr.train_predict(trnX_prime, trnY, tstX_prime, tstY)

	# DPP
	D, V = utils.get_eig(trnL, flag_gpu=flag_gpu)
	E = utils.get_sympoly(D, k, flag_gpu=flag_gpu)
	dpp_smpl  = dpp.sample(D, V, E=E, k=k, flag_gpu=flag_gpu)

	C = L[np.ix_(range(nTrn+nTst), dpp_smpl)]
	W = C[np.ix_(dpp_smpl, range(k))]
	X_prime = C.dot(inv(np.real(sqrtm(W))))
	trnX_prime = X_prime[:nTrn]
	tstX_prime = X_prime[nTrn:]

	error_dpp[k_idx] = lr.train_predict(trnX_prime, trnY, tstX_prime, tstY)

plt.figure(figsize=(4,4))
plt.title('Approximate Kernel LR Test Error')
plt.plot(error_unif, label='unif', lw=2)
plt.plot(error_dpp, label='dpp', lw=2)
plt.legend()

plt.savefig('fig/classification', bbox_inches='tight')




