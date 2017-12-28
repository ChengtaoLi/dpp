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

pairwise_dists = squareform(pdist(np.concatenate((trnX, tstX)), 'euclidean'))
L = np.exp(-pairwise_dists ** 2 / 30 ** 2)
trnL = L[:nTrn, :nTrn]

k_group = [10,20,30,50,70,100]
error_unif = np.zeros((2, len(k_group)))
error_dpp = np.zeros((2, len(k_group)))

for run_id in xrange(5):
	for k_idx in xrange(len(k_group)):
		k = k_group[k_idx]
		# Uniform sampling
		unif_smpl = np.random.permutation(nTrn)[:k]

		C = L[np.ix_(range(nTrn+nTst), unif_smpl)]
		W = C[np.ix_(unif_smpl, range(k))]
		X_prime = C.dot(inv(np.real(sqrtm(W))))
		trnX_prime = X_prime[:nTrn]
		tstX_prime = X_prime[nTrn:]
		
		tmp_trn_err, tmp_tst_err = lr.train_predict(trnX_prime, trnY, tstX_prime, tstY)
		error_unif[0, k_idx] += tmp_trn_err
		error_unif[1, k_idx] += tmp_tst_err

		# DPP
		D, V = utils.get_eig(trnL, flag_gpu=flag_gpu)
		E = utils.get_sympoly(D, k, flag_gpu=flag_gpu)
		dpp_smpl  = dpp.sample(D, V, E=E, k=k, flag_gpu=flag_gpu)

		C = L[np.ix_(range(nTrn+nTst), dpp_smpl)]
		W = C[np.ix_(dpp_smpl, range(k))]
		X_prime = C.dot(inv(np.real(sqrtm(W))))
		trnX_prime = X_prime[:nTrn]
		tstX_prime = X_prime[nTrn:]

		tmp_trn_err, tmp_tst_err = lr.train_predict(trnX_prime, trnY, tstX_prime, tstY)
		error_dpp[0, k_idx] += tmp_trn_err
		error_dpp[1, k_idx] += tmp_tst_err

error_unif /= 5.
error_dpp /= 5.

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.title('Approximate Kernel LR Train Error')
plt.plot(k_group, error_unif[0], label='unif', lw=2)
plt.plot(k_group, error_dpp[0], label='dpp', lw=2)

plt.subplot(1,2,2)
plt.title('Approximate Kernel LR Test Error')
plt.plot(k_group, error_unif[1], label='unif', lw=2)
plt.plot(k_group, error_dpp[1], label='dpp', lw=2)
plt.legend()

plt.savefig('fig/classification', bbox_inches='tight')




