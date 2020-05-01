function [result] = calEst(x, b)
numData = length(x);
n = length(b);
result = zeros(numData, 1);
for i = 1:numData
    for j = 1:n
        result(i) = result(i) + b(j) * x(i) ^ (j - 1);
    end
end
end