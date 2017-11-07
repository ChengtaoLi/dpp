import numpy as np
import torch
import time

# DPP sampler
# input:
#   L: numpy 2d array, kernel for DPP
#   k: size of sampled subset
#   flag_gpu: use gpu acceleration

def sample_dpp(D, V, E=None, k=None, flag_gpu=False):
    if k is None:
        # general dpp
        D = D / (1 + D)
        if flag_gpu:
            v_mask = torch.le(torch.rand(D.size()).cuda(), D.float()).unsqueeze(0)
        else:
            v_mask = torch.le(torch.rand(D.size()), D.float()).unsqueeze(0)

        V = torch.masked_select(V, v_mask).view(D.size(0), -1)
        k = V.size(1)
    else:
        # k-dpp
        v_idx = sample_k(D, E, k, flag_gpu=flag_gpu)
        V = torch.index_select(V, dim=1, index=torch.from_numpy(v_idx))

    rst = list()

    for i in xrange(k-1,-1,-1):
        # choose indices
        P = torch.sum(V**2, dim=1)

        row_idx = torch.multinomial(P, 1).squeeze()
        col_idx = torch.multinomial(torch.ne(V[row_idx], 0.).float(), 1).squeeze()

        rst.append(row_idx)

        V_j = V[:,col_idx]
        V[:,col_idx] = V[:,i]
        V[:,i] = 0.

        # update V
        V = V - torch.mm(V_j, (V[row_idx]/V_j[row_idx]))

        # reorthogonalize
        for a in xrange(i):
            for b in xrange(a):
                V[:,a] = V[:,a] - V[:,a] * V[:,b] * V[:,b]
            V[:,a] = V[:,a] / torch.norm(V[:,a])


    rst = np.sort(torch.cat(rst).numpy())

    return rst


def sample_k(D, E, k, flag_gpu=False):
    i = D.size(0)
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



