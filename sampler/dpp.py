import numpy as np
import time
from scipy.linalg import sqrtm, inv

# DPP sampler
# input:
#   L: numpy 2d array, kernel for DPP
#   k: size of sampled subset
#   flag_gpu: use gpu acceleration

def sample_dpp(D, V, E=None, k=None, flag_gpu=False):
    N = D.shape[0]
    if k is None:
        # general dpp
        D = D / (1 + D)
        V = V[:,np.random.rand(N) < D]
        k = V.shape[1]
    else:
        # k-dpp
        v_idx = sample_k(D, E, k, flag_gpu=flag_gpu)
        V = V[:,v_idx]

    rst = list()

    for i in xrange(k-1,-1,-1):
        # choose indices

        P = np.sum(V**2, axis=1)

        row_idx = np.random.choice(range(N), p=P/np.sum(P))
        col_idx = np.nonzero(V[row_idx])[0][0]

        rst.append(row_idx)

        # update V
        V_j = np.copy(V[:,col_idx])
        V = V - np.outer(V_j, V[row_idx]/V_j[row_idx])
        V[:,col_idx] = V[:,i]
        V = V[:,:i]

        # reorthogonalize
        if i > 0:
            V = sym(V)

    rst = np.sort(rst)

    return rst

def sym(X):
    return X.dot(inv(np.real(sqrtm(X.T.dot(X)))))

def sample_k(D, E, k, flag_gpu=False):
    i = D.shape[0]
    remaining = k
    rst = list()

    while remaining > 0:
        if i == remaining:
            marg = 1.
        else:
            marg = D[i-1] * E[remaining-1, i-1] / E[remaining, i]

        if np.random.rand() < marg:
            rst.append(i-1)
            remaining -= 1
        i -= 1

    return np.array(rst)



