function [centers] = initKmeans(data, k, method)
% initialize k cluster centers
% @param data: input data
% @param k: # of clusters
% @return centers: k cluster centers
numOfData = size(data, 1);
centers = zeros(k, 1);
if method == 1
    % randomly pick k data points as cluster centers
    centers = randperm(numOfData, k);
else
    % k-means++
    count = 1;
    centers(count) = randi(numOfData);
    while count < k
        weights = zeros(numOfData, 1);
        for i = 1:numOfData
            minV = 0;
            for j = 1:count
                dist = norm(data(centers(j)), data(i)) ^ 2;
                if (j == 1 || dist < minV)
                    minV = dist;
                end
            end
            weights(i) = dist;
        end
        total = sum(weights);
        weights = weights / total;
        count = count + 1;
        centers(count) = randsample(1:numOfData, 1, true, weights);
    end
end
end