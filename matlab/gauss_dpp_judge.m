%%   judge if prob < u^T A^{-1} u
function [flag, gauss] = gauss_dpp_judge(A, u, prob, lambdaMin, lambdaMax)

%% Gauss Quadrature results
% Gauss                     -> gauss(1)
% Gauss Radau Lower Bound   -> gauss(2)
% Gauss Radau Upper Bound   -> gauss(3)

% Initialization
K = size(A, 1);
gauss = zeros(3,1);
g = 0;
len = u' * u;

% case that the vector is too small
if len < 1e-10
    if prob < 0
        flag = true;
    else
        flag = false;
    end
    
    return;
end

uPrev = zeros(size(u));
uCurr = u / norm(u);
beta = 0;

% GQL Main Iteration
for k = 1:K
    %fprintf('%d ', k);
    tmp = A * uCurr;
    alpha = uCurr' * tmp;
    
    if k == 1
        g = 1 / alpha;
        c = 1;
        delta = alpha;
        deltaU = alpha - lambdaMin;
        deltaL = alpha - lambdaMax;
    else
        g = g + beta^2 * c^2 / (delta * (alpha * delta - beta^2));
        c = c * beta / delta;
        delta = alpha - beta^2 / delta;
        deltaU = alpha - lambdaMin - beta^2 / deltaU;
        deltaL = alpha - lambdaMax - beta^2 / deltaL;
    end
    
    uT = tmp - alpha * uCurr - beta * uPrev;
    beta = norm(uT);
    alphaU = lambdaMin + beta^2 / deltaU;
    alphaL = lambdaMax + beta^2 / deltaL;
    
    uPrev = uCurr;
    uCurr = uT / beta;
    
    gauss(1) = len * g;
    gauss(2) = len * (g + beta^2 * c^2 / (delta * (alphaL * delta - beta^2)));
    gauss(3) = len * (g + beta^2 * c^2 / (delta * (alphaU * delta - beta^2)));
    
    % approximation is exact
    if beta < 1e-10
        if prob < gauss(1)
            flag = true;
        else
            flag = false;
        end
        return;
    end
    
    if prob <= gauss(2)
        flag = true;
        return;
    elseif prob > gauss(3)
        flag = false;  
        return;
    end
end

% full iterations done, break the tie approximately
kappa = lambdaMax / lambdaMin;
if prob < gauss(2)*kappa/(kappa+1) + gauss(3)/(kappa+1)
    flag = true;
else
    flag = false;
end