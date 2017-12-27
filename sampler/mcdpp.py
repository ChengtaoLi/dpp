import numpy as np
import time
import quadrature
import utils

# MCDPP sampler
# input:
#   L: numpy 2d array, kernel for DPP
#   mix_step: number of mixing steps for Markov chain
#   k: size of sampled subset
#   init_rst: initialization
#   flag_gpu: use gpu acceleration

def sample(L, mix_step, k=None, init_rst=None, flag_gpu=False):
    N = L.shape[0]
    rst = init_rst
    tic_len = mix_step // 10

    if k is None:
        # general dpp
        if rst is None:
            rst = np.random.permutation(N)[:N//3]

        A = np.copy(L[np.ix_(rst, rst)])

        for i in xrange(mix_step):
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))

            u = np.random.randint(N)
            bu = np.copy(L[np.ix_([u], rst)])
            cu = np.copy(L[np.ix_([u], [u])])

            ind = np.where(rst == u)[0]
            lambda_min, lambda_max = utils.gershgorin(A)
            lambda_min = np.max([lambda_min, 1e-5])

            if ind.size == 0: # try to add
                # print('adding...')
                flag = quadrature.gauss_dpp_judge(A, bu[0], cu[0,0]-np.random.uniform(), lambda_min, lambda_max)
                if not flag:
                    rst = np.append(rst, [u])
                    A = np.r_[np.c_[A, bu.transpose()], np.c_[bu, cu]]
                    
            else: # try to remove
                ind = ind[0]
                tmp_rst = np.copy(rst)
                tmp_rst = np.delete(tmp_rst, ind)
                tmp_A = np.copy(A)
                tmp_A = np.delete(tmp_A, ind, axis=0)
                tmp_A = np.delete(tmp_A, ind, axis=1)
                bu = np.delete(bu, ind, axis=1)

                flag = quadrature.gauss_dpp_judge(tmp_A, bu[0], cu[0,0]-1/np.random.uniform(), lambda_min, lambda_max)
                if flag:
                    rst = tmp_rst
                    A = tmp_A
                
    else:
        # k-dpp
        if rst is None:
            rst = rst = np.random.permutation(N)[:k]
        rst_bar = np.setdiff1d(range(N), rst)

        A = np.copy(L[np.ix_(rst, rst)])

        for i in xrange(mix_step):
            if (i+1) % tic_len == 0:
                print('{}-th iteration.'.format(i+1))
            rem_ind = np.random.randint(k)
            add_ind = np.random.randint(N-k)
            v = rst[rem_ind]
            u = rst_bar[add_ind]

            tmp_rst = np.delete(np.copy(rst), rem_ind)
            tmp_rst_bar = np.delete(np.copy(rst_bar), add_ind)
            tmp_A = np.copy(A)
            tmp_A = np.delete(tmp_A, rem_ind, axis=0)
            tmp_A = np.delete(tmp_A, rem_ind, axis=1)
            bu = np.copy(L[np.ix_([u], tmp_rst)])
            bv = np.copy(L[np.ix_([v], tmp_rst)])

            lambda_min, lambda_max = utils.gershgorin(tmp_A)
            lambda_min = np.max([lambda_min, 1e-5])

            prob = np.random.uniform()
            tar = prob * L[v,v] - L[u,u]

            flag = quadrature.gauss_kdpp_judge(tmp_A, bu[0], bv[0], prob, tar, lambda_min, lambda_max)

            if flag:
                rst = np.append(tmp_rst, [u])
                rst_bar = np.append(tmp_rst_bar, [v])
                A = np.r_[np.c_[tmp_A, bu.transpose()], np.c_[bu, L[u,u]]]

    return rst




