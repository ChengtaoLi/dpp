function [lambdaMin, lambdaMax] = gershgorin(A)

radius = full(sum(abs(A)));

lambdaMax = max(radius);
lambdaMin = min(2 * diag(A) - radius');