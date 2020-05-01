function [x, result] = genDataPoint(var, w)
lb = -1;
ub = 1;
x = lb + (ub - lb) * rand;
error = genGaussian(0, var);

phi = zeros(1, length(w));
for i = 1:length(phi)
    phi(i) = x ^ (i - 1);
end

result = phi * w + error;
end

