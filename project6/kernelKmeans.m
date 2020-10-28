function [curr, count] = kernelKmeans(data, gram, k, height, width, filename)
points = initKmeans(data, k, 1);
numOfData = size(data, 1);
curr = zeros(numOfData, 1);
c = 1;
for i = 1:length(points)
    curr(points(i)) = c;
    c = c + 1;
end
count = 0;
while 1
    prev = curr;
    % kernel k-means algorithm
    clusters = cell(k, 1);
    squareTerms = zeros(k, 1);
    % pre-compute the last term in the distance formula to speed up
    for class = 1:k
        members = find(prev == class);
        clusters{class} = members;
        clusterSize = size(members, 1);
        for m = 1:clusterSize
            for n = 1:clusterSize
                squareTerms(class) = squareTerms(class) + gram(members(m), members(n)); 
            end
        end
        squareTerms(class) = squareTerms(class) / (clusterSize^2);
    end
    for i = 1:numOfData
        minI = 0;
        minV = 0;
        for j = 1:k
            mid = 0;
            members = clusters{j};
            clusterSize = size(members, 1);
            for m = 1:clusterSize
                mid = mid + gram(i, members(m));
            end
            value = gram(i, i) - (2 / clusterSize) * mid + squareTerms(j);
            if (minI == 0 || value < minV)
                minI = j;
                minV = value;
            end
        end
        curr(i) = minI;
    end
    % generate the cluster image
    generateClusterImage(curr, height, width, filename, count);
    % there is no change in cluster assignment
    if prev == curr
        break;
    end
    count = count + 1;
end

end