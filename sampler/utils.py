import numpy as np
import torch

def get_eig(L, flag_gpu=False):
    if flag_gpu:
        L = torch.from_numpy(L).cuda()
    else:
        L = torch.from_numpy(L)

    return torch.symeig(L, eigenvectors=True)

def get_sympoly(D, k, flag_gpu=False):
    N = D.size(0)
    if flag_gpu:
        E = torch.zeros(k+1, N+1).float().cuda()
    else:
        E = torch.zeros(k+1, N+1).float()

    E[0] = 1.
    for l in xrange(1,k+1):
        E[l,1:] = D.float().unsqueeze(0) * E[l-1,:N].unsqueeze(0)
        torch.cumsum(E[l], dim=0, out=E[l])

    return E



