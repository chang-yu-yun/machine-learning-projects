function [curr, count, means] = kmeans(data, k, height, width, filename)
points = initKmeans(data, k, 1);
numOfData = size(data, 1);
dim = size(data, 2);
means = zeros(k, dim);
for i = 1:k
    means(i, :) = data(points(i), :);
end
curr = zeros(numOfData, 1);
count = 0;
while 1
    prev = curr;
    % E step
    for i = 1:numOfData
        minI = 0;
        minV = 0;
        for j = 1:k
            currV = norm(data(i, :) - means(j, :)) ^ 2;
            if (minI == 0 || currV < minV)
                minI = j;
                minV = currV;
            end
        end
        curr(i) = minI;
    end
    
    % generate cluster image to visualize partitioning of data points
    % generateClusterImage(curr, height, width, filename, count);
    
    % M step
    means = zeros(k, dim);
    for class = 1:k
        members = find(curr == class);
        classSize = size(members, 1);
        for j = 1:classSize
            means(class, :) = means(class, :) + data(members(j), :);
        end
        means(class, :) = means(class, :) ./ classSize;
    end
    
    % there is no change in cluster assignment
    if prev == curr
        break;
    end
    count = count + 1;
end
end