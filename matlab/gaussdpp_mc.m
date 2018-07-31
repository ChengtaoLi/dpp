%% sampling subsets from (Gibbs) Markov chain k-DPP with Gauss quadrature
%
% -input
%   L: data kernel matrix, N*N where N is number of samples
%   rangeFun: range function for bounding eigenspectrum of matrices
%   mixStep: number of burn-in iterations
%   k: the size of sampled subset
%   init_C: initialization, should be of size k
%
% -output
%   C: sampled subset
%
% sample usage:
%   C = gaussdpp_mc(L,@gershgorin,1000,5,[1,2,3,4,5])

function C = gaussdpp_mc(L, rangeFun, mixStep, k, init_C)
    C = gauss_kdpp(L, k, rangeFun, mixStep, init_C);
    C = sort(C, 'ascend');
end


function C = gauss_kdpp(L, k, rangeFun, mixStep, init_C)
    n = length(L);
    C = init_C;
    A = L(C,C);
    
    for i = 1:mixStep
        delInd = randi(k);
        v = C(delInd); % one to remove
        u = randi(n); % one to add
        while any(C == u)
            u = randi(n);
        end
        tmpC = C; tmpA = A;
        tmpC(delInd) = []; tmpA(delInd,:) = []; tmpA(:,delInd) = [];
        bu = L(tmpC, u); bv = L(tmpC, v);
        Luu = L(u,u);
        Lvv = L(v,v);
        
        [lambdaMin, lambdaMax] = rangeFun(tmpA);
        lambdaMin = max(lambdaMin, 1e-5);
            
        prob = rand;
        prob = prob / (1 - prob);
        tar = full(prob * Lvv - Luu);
        flag = gauss_kdpp_judge(tmpA, bu, bv, prob, tar, lambdaMin, lambdaMax);
        if flag % accept move
            C = [tmpC u];
            A = [tmpA bu; bu' Luu];
        end
    end
end



