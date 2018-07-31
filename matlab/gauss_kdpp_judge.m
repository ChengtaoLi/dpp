%%   judge if tar < prob * v' * inv(L) * v - u' * inv(L) * u
function [flag] = gauss_kdpp_judge(A, u, v, prob, tar, lambdaMin, lambdaMax)

%% Gauss Quadrature results
% Gauss                     -> gauss_U(1) and gauss_V(1)
% Gauss Radau Lower Bound   -> gauss_U(2) and gauss_V(2)
% Gauss Radau Upper Bound   -> gauss_U(3) and gauss_V(3)

% Initialization
K = size(A, 1);
gauss_U = zeros(3,1);
gauss_V = zeros(3,1);
g_U = 0;
g_V = 0;
len_U = u' * u;
len_V = v' * v;

% case that the vector is too small
if len_U < 1e-10
    [flag,~] = gauss_dpp_judge(A, v, tar/prob, lambdaMin, lambdaMax);
    return;
    
elseif len_V < 1e-10
    [flag,~] = gauss_dpp_judge(A, u, -tar, lambdaMin, lambdaMax);
    flag = ~flag;
    return;
end

prev_U = zeros(size(u));
prev_V = zeros(size(v));
curr_U = u / norm(u);
curr_V = v / norm(v);
beta_U = 0;
beta_V = 0;
c_U = 1;
c_V = 1;

g_U = 0;
g_V = 0;

delta_U = 0;
deltaU_U = 0;
deltaL_U = 0;
delta_V = 0;
deltaU_V = 0;
deltaL_V = 0;

alpha_U = 0;
alphaU_U = 0;
alphaL_U = 0;
alpha_V = 0;
alphaU_V = 0;
alphaL_V = 0;

iter_U = 1;
iter_V = 1;
gap_U = -1;
gap_V = -1;

proceed_u();
proceed_v();

if tar <= prob * gauss_V(2) - gauss_U(3)
    flag = true;
    return;
elseif tar >= prob * gauss_V(3) - gauss_U(2)
    flag = false;
    return;
end
    

while iter_U < K && iter_V < K
    if gap_U >= prob * gap_V
        proceed_u();
    else
        proceed_v();
    end
    
    if tar <= prob * gauss_V(2) - gauss_U(3)
        flag = true;
        return;
    elseif tar >= prob * gauss_V(3) - gauss_U(2)
        flag = false;
        return;
    end
    
end

while iter_U < K
    proceed_u();
    if tar <= prob * gauss_V(2) - gauss_U(3)
        flag = true;
        return;
    elseif tar >= prob * gauss_V(3) - gauss_U(2)
        flag = false;
        return;
    end
end

while iter_V < K
    proceed_v();
    if tar <= prob * gauss_V(2) - gauss_U(3)
        flag = true;
        return;
    elseif tar >= prob * gauss_V(3) - gauss_U(2)
        flag = false;
        return;
    end
end

% full iterations done, break the tie approximately
if tar <= (prob * (gauss_V(2) + gauss_V(3)) - (gauss_U(2) + gauss_U(3))) / 2
    flag = true;
else
    flag = false;
end

return;


function proceed_u()
    tmp = A * curr_U;
    alpha_U = curr_U' * tmp;
    
    if iter_U == 1
        g_U = 1 / alpha_U;
        c_U = 1;
        delta_U = alpha_U;
        deltaU_U = alpha_U - lambdaMin;
        deltaL_U = alpha_U - lambdaMax;
    else
        g_U = g_U + beta_U^2 * c_U^2 / (delta_U * (alpha_U * delta_U - beta_U^2));
        c_U = c_U * beta_U / delta_U;
        delta_U = alpha_U - beta_U^2 / delta_U;
        deltaU_U = alpha_U - lambdaMin - beta_U^2 / deltaU_U;
        deltaL_U = alpha_U - lambdaMax - beta_U^2 / deltaL_U;
    end
    
    uT = tmp - alpha_U * curr_U - beta_U * prev_U;
    beta_U = norm(uT);
    alphaU_U = lambdaMin + beta_U^2 / deltaU_U;
    alphaL_U = lambdaMax + beta_U^2 / deltaL_U;
    
    prev_U = curr_U;
    curr_U = uT / beta_U;
    
    gauss_U(1) = len_U * g_U;
    gauss_U(2) = len_U * (g_U + beta_U^2 * c_U^2 / (delta_U * (alphaL_U * delta_U - beta_U^2)));
    gauss_U(3) = len_U * (g_U + beta_U^2 * c_U^2 / (delta_U * (alphaU_U * delta_U - beta_U^2)));
    gap_U = gauss_U(3) - gauss_V(2);
    
    iter_U = iter_U + 1;
    
    if beta_U < 1e-5
        iter_U = K
    end
end

function proceed_v()
    tmp = A * curr_V;
    alpha_V = curr_V' * tmp;
    
    if iter_V == 1
        g_V = 1 / alpha_V;
        c_V = 1;
        delta_V = alpha_V;
        deltaU_V = alpha_V - lambdaMin;
        deltaL_V = alpha_V - lambdaMax;
    else
        g_V = g_V + beta_V^2 * c_V^2 / (delta_V * (alpha_V * delta_V - beta_V^2));
        c_V = c_V * beta_V / delta_V;
        delta_V = alpha_V - beta_V^2 / delta_V;
        deltaU_V = alpha_V - lambdaMin - beta_V^2 / deltaU_V;
        deltaL_V = alpha_V - lambdaMax - beta_V^2 / deltaL_V;
    end
    
    uT = tmp - alpha_V * curr_V - beta_V * prev_V;
    beta_V = norm(uT);
    alphaU_V = lambdaMin + beta_V^2 / deltaU_V;
    alphaL_V = lambdaMax + beta_V^2 / deltaL_V;
    
    prev_V = curr_V;
    curr_V = uT / beta_V;
    
    gauss_V(1) = len_V * g_V;
    gauss_V(2) = len_V * (g_V + beta_V^2 * c_V^2 / (delta_V * (alphaL_V * delta_V - beta_V^2)));
    gauss_V(3) = len_V * (g_V + beta_V^2 * c_V^2 / (delta_V * (alphaU_V * delta_V - beta_V^2)));
    gap_V = gauss_V(3) - gauss_V(2);
    
    iter_V = iter_V + 1;
    
    if beta_V < 1e-5
        iter_V = K
    end
end
end