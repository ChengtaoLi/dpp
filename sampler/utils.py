import numpy as np
import scipy

import gzip
import os
from os import path

import sys
if sys.version_info.major < 3:
    import urllib
else:
    import urllib.request as request

DATASET_DIR = 'data/'

MNIST_FILES = ["train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
               "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"]

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



# MNIST data processor

def download_file(url, local_path):
    dir_path = path.dirname(local_path)
    if not path.exists(dir_path):
        print("Creating the directory '%s' ..." % dir_path)
        os.makedirs(dir_path)

    print("Downloading from '%s' ..." % url)
    if sys.version_info.major < 3:
        urllib.URLopener().retrieve(url, local_path)
    else:
        request.urlretrieve(url, local_path)


def download_mnist(local_path):
    url_root = "http://yann.lecun.com/exdb/mnist/"
    for f_name in MNIST_FILES:
        f_path = os.path.join(local_path, f_name)
        if not path.exists(f_path):
            download_file(url_root + f_name, f_path)


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def load_mnist(ntrain=4000, ntest=1000, onehot=True):
    data_dir = os.path.join(DATASET_DIR, 'mnist/')
    if not path.exists(data_dir):
        download_mnist(data_dir)
    else:
        # check all files
        checks = [path.exists(os.path.join(data_dir, f)) for f in MNIST_FILES]
        if not np.all(checks):
            download_mnist(data_dir)

    with gzip.open(os.path.join(data_dir, 'train-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        trY = loaded[8:].reshape((60000))

    with gzip.open(os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    with gzip.open(os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')) as fd:
        buf = fd.read()
        loaded = np.frombuffer(buf, dtype=np.uint8)
        teY = loaded[8:].reshape((10000))

    trX /= 255.
    teX /= 255.

    trPerm = np.random.permutation(60000)
    trX = trX[trPerm][:ntrain]
    trY = trY[trPerm][:ntrain]

    tePerm = np.random.permutation(10000)
    teX = teX[tePerm][:ntest]
    teY = teY[tePerm][:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY
