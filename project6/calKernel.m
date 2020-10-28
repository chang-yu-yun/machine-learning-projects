function [gram] = calKernel(data, width, gammaS, gammaC)
% kernel function: k(x, x') = exp(-gammaS * ||S(x) - S(x')|| ** 2) * exp(-gammaC * ||C(x) - C(x')|| ** 2)
numOfData = size(data, 1);
gram = zeros(numOfData, numOfData);
for i = 1:numOfData
    for j = 1:numOfData
        s1 = [floor((i - 1) / width), mod(i - 1, width)];
        s2 = [floor((j - 1) / width), mod(j - 1, width)];
        c1 = data(i, :);
        c2 = data(j, :);
        gram(i, j) = exp(-gammaS * norm(s1 - s2) ^ 2) * exp(-gammaC * norm(c1 - c2) ^ 2);
    end
end
end