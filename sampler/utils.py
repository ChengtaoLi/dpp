import numpy as np
import scipy

def get_eig(L, flag_gpu=False):
    if flag_gpu:
        pass
    else:
        return scipy.linalg.eigh(L)

def get_sympoly(D, k, flag_gpu=False):
    N = D.shape[0]
    if flag_gpu:
        pass
    else:
        E = np.zeros((k+1, N+1))

    E[0] = 1.
    for l in xrange(1,k+1):
        E[l,1:] = np.copy(np.multiply(D, E[l-1,:N]))
        E[l] = np.cumsum(E[l], axis=0)

    return E


def gershgorin(A):
    radius = np.sum(A, axis=0)
    
    lambda_max = np.max(radius)
    lambda_min = np.min(2 * np.diag(A) - radius)

    return lambda_max, lambda_min
