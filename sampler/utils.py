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


def kpp(X, k, flag_kernel=False):
    # if X is not kernel, rows of X are samples

    N = X.shape[0]
    rst = np.zeros(k, dtype=int)
    rst[0] = np.random.randint(N)

    if flag_kernel:
        # kernel kmeans++
        v = np.ones(N) * np.inf
        for i in xrange(1, k):
            Y = np.diag(X) + np.ones(N)*X[rst[i-1],rst[i-1]] - 2*X[rst[i-1]]
            v = np.minimum(v,Y)
            r = np.random.uniform()
            rst[i] = np.where(v.cumsum() / v.sum() >= r)[0][0]

    else:
        # normal kmeans++
        centers = [X[rst[0]]]
        for i in xrange(1, k):
            dist = np.array([min([np.linalg.norm(x-c)**2 for c in centers]) for x in X])
            r = np.random.uniform()
            ind = np.where(dist.cumsum() / dist.sum() >= r)[0][0]
            rst[i] = ind
            centers.append(X[ind])

    return rst
