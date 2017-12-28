import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import inv, norm

import sampler.dpp as dpp
import sampler.utils as utils

# currently only support cpu mode
flag_gpu = False

lmbd = 1e-4
sigma = 100
data = np.loadtxt('data/ailerons.txt')[:5000]
nTrn = 4000
trnY = data[:nTrn, -1]
tstY = data[nTrn:, -1]
pairwise_dists = squareform(pdist(data[:,:-1], 'euclidean'))
L = np.exp(-pairwise_dists ** 2 / 100 ** 2)
trnL = L[:nTrn, :nTrn]

k_group = [20,30,50,70,100]
error_unif = np.zeros((3, len(k_group)))
error_dpp = np.zeros((3, len(k_group)))

for k_idx in xrange(len(k_group)):
	k = k_group[k_idx]
	# Uniform sampling
	unif_smpl = np.random.permutation(nTrn)[:k]

	C = trnL[np.ix_(range(nTrn), unif_smpl)]
	W = C[np.ix_(unif_smpl, range(k))]
	trnL_prime = C.dot(inv(W)).dot(C.transpose())
	error_unif[0,k_idx] = norm(trnL_prime - trnL, 'fro')
	alpha = inv(trnL_prime + nTrn * lmbd * np.identity(nTrn)).dot(trnY)
	Y_hat = L[:,:nTrn].dot(alpha)
	error_unif[1,k_idx] = norm(Y_hat[:nTrn] - trnY)
	error_unif[2,k_idx] = norm(Y_hat[nTrn:] - tstY)

	# DPP
	D, V = utils.get_eig(trnL, flag_gpu=flag_gpu)
	E = utils.get_sympoly(D, k, flag_gpu=flag_gpu)
	dpp_smpl  = dpp.sample(D, V, E=E, k=k, flag_gpu=flag_gpu)

	C = trnL[np.ix_(range(nTrn), dpp_smpl)]
	W = C[np.ix_(dpp_smpl, range(k))]
	trnL_prime = C.dot(inv(W)).dot(C.transpose())
	error_dpp[0,k_idx] = norm(trnL_prime - trnL, 'fro')
	alpha = inv(trnL_prime + nTrn * lmbd * np.identity(nTrn)).dot(trnY)
	Y_hat = L[:,:nTrn].dot(alpha)
	error_dpp[1,k_idx] = norm(Y_hat[:nTrn] - trnY)
	error_dpp[2,k_idx] = norm(Y_hat[nTrn:] - tstY)

plt.figure(figsize=(12,4))
plt.subplot(1,3,1)
plt.title('Nystrom Approximation Error')
plt.plot(error_unif[0], label='unif', lw=2)
plt.plot(error_dpp[0], label='dpp', lw=2)
plt.subplot(1,3,2)
plt.title('Approximate KRR Train Error')
plt.plot(error_unif[1], label='unif', lw=2)
plt.plot(error_dpp[1], label='dpp', lw=2)
plt.subplot(1,3,3)
plt.title('Approximate KRR Test Error')
plt.plot(error_unif[2], label='unif', lw=2)
plt.plot(error_dpp[2], label='dpp', lw=2)
plt.legend()

plt.savefig('fig/regression', bbox_inches='tight')




