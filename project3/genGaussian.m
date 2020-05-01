function [result] = genGaussian(mean, var)
std = sqrt(var);
% u and v are two independent uniform-distributed random variables
% Unif(0, 1)
u = rand;
rng('shuffle');
v = rand;

s = sqrt((-2) * log(u)) * cos(2 * pi * v);
result = std * s + mean;
end

