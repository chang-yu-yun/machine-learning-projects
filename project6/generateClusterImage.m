function [] = generateClusterImage(clusters, height, width, filename, append)
colormap = [[255, 0, 0]; [0, 255, 0]; [0, 0, 255]; [0, 255, 255]];
numOfData = size(clusters, 1);
clusterImage = zeros(numOfData, 3);
for i = 1:numOfData
    clusterImage(i, :) = colormap(clusters(i), :);
end
image = zeros(height, width, 3);
for i = 1:height
    for j = 1:width
        image(i, j, :) = clusterImage((i - 1) * width + j, :);
    end
end
[currCluster, map] = rgb2ind(image, 256);
if append == 0
    imwrite(currCluster, map, filename, 'gif');
else
    imwrite(currCluster, map, filename, 'gif', 'WriteMode', 'append');
end
end