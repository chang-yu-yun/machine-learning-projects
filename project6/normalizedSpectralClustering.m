function [clusters, count, means] = normalizedSpectralClustering(W, k, height, width, filename)
% unnormalized spectral clustering
% @param W: similarity matrix
% @param k: # of clusters
% @return clusters: cluster number for each data point
% @return count: # of iteration for convergence
numOfData = size(W, 1);
D = zeros(numOfData, numOfData);
for i = 1:numOfData
    D(i, i) = sum(W(i, :));
end
L = D - W;
Q = D^(-1/2);
Lsym = Q * L * Q;
[eigenvectors, eigenvalues] = eig(Lsym);
% val: the diagonal elements are eigenvalues
% vec: the columns are the corresponding eigenvectors
[d, ind] = sort(diag(eigenvalues));
eigenvalues = eigenvalues(ind, ind);
eigenvectors = eigenvectors(:, ind);
U = eigenvectors(:, 2:(k + 1));
for i = 1:numOfData
    len = norm(U(i, :));
    U(i, :) = U(i, :) ./ len;
end
[clusters, count, means] = kmeans(U, k, height, width, filename);
if k == 3
    figure;
    for i = 1:3
        target = find(clusters == i);
        clusterSize = length(target);
        points = zeros(clusterSize, 3);
        for j = 1:clusterSize
            points(j, :) = U(target(j), :);
        end
        plot3(points(:, 1), points(:, 2), points(:, 3), 'o', 'MarkerSize', 8);
        hold on;
    end
    grid on;
end
end