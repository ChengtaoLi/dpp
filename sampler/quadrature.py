import numpy as np
from scipy.linalg import inv

# function judging if prob < u^T A^{-1} u
def gauss_dpp_judge(A, u, prob, lambda_min, lambda_max):

    # Gauss Quadrature results
    # Gauss                     -> gauss(1,:)
    # Gauss Radau Lower Bound   -> gauss(2,:)
    # Gauss Radau Upper Bound   -> gauss(3,:)
    # Gauss Lobatto             -> gauss(4,:)

    # Initialization
    K = A.shape[0]
    gauss = np.zeros(4)

    u_len = np.inner(u,u)

    # case that the vector is too small
    if u_len < 1e-10:
        if prob < 0:
            return True
        else:
            return False

    g = 0
    p = np.copy(u)
    beta = 0
    gamma = 1
    c = 1

    f, fU, fL, fT = 0, 0, 0, 0
    delta, deltaU, deltaL = 0, 0, 0
    eta, etaT = 0, 0
    alpha, alphaU, alphaL, alphaT = 0, 0, 0, 0

    truth = np.inner(u, inv(A).dot(u));

    # CGQL Main Iteration
    for k in xrange(2*K):
        #('--k: {}'.format(k))

        newGamma = np.inner(u, u) / np.inner(p, A.dot(p))
        alpha = 1 / newGamma + beta / gamma
        gamma = newGamma
        
        if k == 0:
            f = 1 / alpha
            delta = alpha
            deltaU = alpha - lambda_min
            deltaL = alpha - lambda_max
        else:
            c = c * eta / (delta**2)
            delta = 1 / gamma
            f = gamma * c
            deltaU = alpha - alphaU
            deltaL = alpha - alphaL
        
        beta = np.inner(u, u)
        u = u - gamma * A.dot(p)
        beta = np.inner(u,u) / beta
        eta = beta / (gamma**2)
        p = u + beta * p
        
        alphaU = lambda_min + eta / deltaU
        alphaL = lambda_max + eta / deltaL
        alphaT = deltaU * deltaL / (deltaL - deltaU)
        etaT = alphaT * (lambda_max - lambda_min)
        alphaT = alphaT * (lambda_max / deltaU - lambda_min / deltaL)

        fU = eta * c / (delta * (alphaU * delta - eta)) 
        fL = eta * c / (delta * (alphaL * delta - eta))
        fT = etaT * c / (delta * (alphaT * delta - etaT))
        
        g = g + f
        gauss[0] = u_len * g
        gauss[1] = u_len * (g + fL)
        gauss[2] = u_len * (g + fU)
        gauss[3] = u_len * (g + fT)
        
        # print('prob: {}, trutu: {}, l1: {}, l2: {}, u1: {}, u2: {}'.format( \
        #     prob, truth, gauss[0], gauss[1], gauss[2], gauss[3]))
        
        # approximation is exact
        if eta < 1e-10:
            if prob < gauss[0]:
                return True
            else:
                return False
        
        if prob < max(gauss[:2]):
            return True
        elif prob > min(gauss[2:]):
            return False
        
    if prob < max(gauss[:2]) + min(gauss[2:]) / 2:
        return True
    else:
        return False



# judge if tar < prob * v' * inv(L) * v - u' * inv(L) * u
def gauss_kdpp_judge(A, u, v, prob, tar, lambda_min, lambda_max):

    # Gauss Quadrature results
    # Gauss                     -> gauss[0]
    # Gauss Radau Lower Bound   -> gauss[1]
    # Gauss Radau Upper Bound   -> gauss[2]
    # Gauss Lobatto             -> gauss[3]

    # Initialization
    K = A.shape[0]
    gauss_U, gauss_V = np.zeros(4), np.zeros(4)
    g_U, g_V = 0, 0
    len_U, len_V = np.inner(u,u), np.inner(v,v)

    # case that the vector is too small
    if len_U < 1e-10:
        return gauss_dpp_judge(A, v, tar/prob, lambdaMin, lambdaMax);
        
    elif len_V < 1e-10:
        return not gauss_dpp_judge(A, u, -tar, lambdaMin, lambdaMax);

    p_U = np.copy(u)
    p_V = np.copy(v)
    beta_U, beta_V = 0, 0
    gamma_U, gamma_V = 1, 1
    c_U, c_V = 1, 1

    f_U, fU_U, fL_U, fT_U = 0, 0, 0, 0
    f_V, fU_V, fL_V, fT_V = 0, 0, 0, 0

    delta_U, deltaU_U, deltaL_U = 0, 0, 0
    delta_V, deltaU_V, deltaL_V = 0, 0, 0

    eta_U, etaT_U = 0, 0
    eta_V, etaT_V = 0, 0

    alpha_U, alphaU_U, alphaL_U, alphaT_U = 0, 0, 0, 0
    alpha_V, alphaU_V, alphaL_V, alphaT_V = 0, 0, 0, 0

    iter_U = 1
    iter_V = 1
    gap_U = -1
    gap_V = -1

    # proceed_u():
    newGamma_U = np.inner(u, u) / np.inner(p_U, A.dot(p_U))
    alpha_U = 1 / newGamma_U + beta_U / gamma_U
    gamma_U = newGamma_U
    
    f_U = 1 / alpha_U
    delta_U = alpha_U
    deltaU_U = alpha_U - lambda_min
    deltaL_U = alpha_U - lambda_max
    
    beta_U = np.inner(u, u)
    u = u - gamma_U * A.dot(p_U)
    beta_U = np.inner(u, u) / beta_U
    eta_U = beta_U / (gamma_U**2)
    p_U = u + beta_U * p_U
    
    alphaU_U = lambda_min + eta_U / deltaU_U
    alphaL_U = lambda_max + eta_U / deltaL_U
    alphaT_U = deltaU_U * deltaL_U / (deltaL_U - deltaU_U)
    etaT_U = alphaT_U * (lambda_max - lambda_min)
    alphaT_U = alphaT_U * (lambda_max / deltaU_U - lambda_min / deltaL_U)
    
    fU_U = eta_U * c_U / (delta_U * (alphaU_U * delta_U - eta_U))
    fL_U = eta_U * c_U / (delta_U * (alphaL_U * delta_U - eta_U))
    fT_U = etaT_U * c_U / (delta_U * (alphaT_U * delta_U - etaT_U))
    
    g_U = g_U + f_U
    gauss_U[0] = len_U * g_U
    gauss_U[1] = len_U * (g_U + fL_U)
    gauss_U[2] = len_U * (g_U + fU_U)
    gauss_U[3] = len_U * (g_U + fT_U)
    gap_U = np.min(gauss_U[2:]) - np.max(gauss_U[:2])
        

    # proceed_v():
    newGamma_V = np.inner(v, v) / np.inner(p_V, A.dot(p_V))
    alpha_V = 1 / newGamma_V + beta_V / gamma_V
    gamma_V = newGamma_V
    
    f_V = 1 / alpha_V
    delta_V = alpha_V
    deltaU_V = alpha_V - lambda_min
    deltaL_V = alpha_V - lambda_max
    
    beta_V = np.inner(v, v)
    v = v - gamma_V * A.dot(p_V)
    beta_V = np.inner(v, v) / beta_V
    eta_V = beta_V / (gamma_V**2)
    p_V = v + beta_V * p_V
    
    alphaU_V = lambda_min + eta_V / deltaU_V
    alphaL_V = lambda_max + eta_V / deltaL_V
    alphaT_V = deltaU_V * deltaL_V / (deltaL_V - deltaU_V)
    etaT_V = alphaT_V * (lambda_max - lambda_min)
    alphaT_V = alphaT_V * (lambda_max / deltaU_V - lambda_min / deltaL_V)
    
    fU_V = eta_V * c_V / (delta_V * (alphaU_V * delta_V - eta_V))
    fL_V = eta_V * c_V / (delta_V * (alphaL_V * delta_V - eta_V))
    fT_V = etaT_V * c_V / (delta_V * (alphaT_V * delta_V - etaT_V))
    
    g_V = g_V + f_V
    gauss_V[0] = len_V * g_V
    gauss_V[1] = len_V * (g_V + fL_V)
    gauss_V[2] = len_V * (g_V + fU_V)
    gauss_V[3] = len_V * (g_V + fT_V)
    gap_V = np.min(gauss_V[2:]) - np.max(gauss_V[:2])

    # debug_info();
    if tar <= prob * np.max(gauss_V[:2]) - np.min(gauss_U[2:]):
        return True
    elif tar >= prob * np.min(gauss_V[2:]) - np.max(gauss_U[:2]):
        return False

    for k in xrange(4*K):
        if gap_U >= prob * gap_V:
            newGamma_U = np.inner(u, u) / np.inner(p_U, A.dot(p_U))
            alpha_U = 1 / newGamma_U + beta_U / gamma_U
            gamma_U = newGamma_U
            
            c_U = c_U * eta_U / (delta_U**2)
            delta_U = 1 / gamma_U
            f_U = gamma_U * c_U
            deltaU_U = alpha_U - alphaU_U
            deltaL_U = alpha_U - alphaL_U
            
            beta_U = np.inner(u, u)
            u = u - gamma_U * A.dot(p_U)
            beta_U = np.inner(u, u) / beta_U
            eta_U = beta_U / (gamma_U**2)
            p_U = u + beta_U * p_U
            
            alphaU_U = lambda_min + eta_U / deltaU_U
            alphaL_U = lambda_max + eta_U / deltaL_U
            alphaT_U = deltaU_U * deltaL_U / (deltaL_U - deltaU_U)
            etaT_U = alphaT_U * (lambda_max - lambda_min)
            alphaT_U = alphaT_U * (lambda_max / deltaU_U - lambda_min / deltaL_U)
            
            fU_U = eta_U * c_U / (delta_U * (alphaU_U * delta_U - eta_U))
            fL_U = eta_U * c_U / (delta_U * (alphaL_U * delta_U - eta_U))
            fT_U = etaT_U * c_U / (delta_U * (alphaT_U * delta_U - etaT_U))
            
            g_U = g_U + f_U
            gauss_U[0] = len_U * g_U
            gauss_U[1] = len_U * (g_U + fL_U)
            gauss_U[2] = len_U * (g_U + fU_U)
            gauss_U[3] = len_U * (g_U + fT_U)
            gap_U = np.min(gauss_U[2:]) - np.max(gauss_U[:2])
            
            iter_U = iter_U + 1

        else:
            newGamma_V = np.inner(v, v) / np.inner(p_V, A.dot(p_V))
            alpha_V = 1 / newGamma_V + beta_V / gamma_V
            gamma_V = newGamma_V
            
            c_V = c_V * eta_V / (delta_V**2)
            delta_V = 1 / gamma_V
            f_V = gamma_V * c_V
            deltaU_V = alpha_V - alphaU_V
            deltaL_V = alpha_V - alphaL_V
            
            beta_V = np.inner(v, v)
            v = v - gamma_V * A.dot(p_V)
            beta_V = np.inner(v, v) / beta_V
            eta_V = beta_V / (gamma_V**2)
            p_V = v + beta_V * p_V
            
            alphaU_V = lambda_min + eta_V / deltaU_V
            alphaL_V = lambda_max + eta_V / deltaL_V
            alphaT_V = deltaU_V * deltaL_V / (deltaL_V - deltaU_V)
            etaT_V = alphaT_V * (lambda_max - lambda_min)
            alphaT_V = alphaT_V * (lambda_max / deltaU_V - lambda_min / deltaL_V)
            
            fU_V = eta_V * c_V / (delta_V * (alphaU_V * delta_V - eta_V))
            fL_V = eta_V * c_V / (delta_V * (alphaL_V * delta_V - eta_V))
            fT_V = etaT_V * c_V / (delta_V * (alphaT_V * delta_V - etaT_V))
            
            g_V = g_V + f_V
            gauss_V[0] = len_V * g_V
            gauss_V[1] = len_V * (g_V + fL_V)
            gauss_V[2] = len_V * (g_V + fU_V)
            gauss_V[3] = len_V * (g_V + fT_V)
            gap_V = np.min(gauss_V[2:]) - np.max(gauss_V[:2])
            
            iter_V = iter_V + 1
        
        if tar <= prob * np.max(gauss_V[:2]) - np.min(gauss_U[2:]):
            return True
        elif tar >= prob * np.min(gauss_V[2:]) - np.max(gauss_U[:2]):
            return False
        