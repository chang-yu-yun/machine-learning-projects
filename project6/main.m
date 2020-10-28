close all;
clear;
clc;
%% read image
for file = 1:2
    filename = ['image' num2str(file) '.png'];
    index = find(filename == '.');
    last = index - 1;
    prefix = filename(1:last);
    image = imread(filename);
    height = size(image, 1);
    width = size(image, 2);
    numOfData = height * width;
    
    data = zeros(numOfData, 3);
    for i = 1:height
        for j = 1:width
            data((i - 1) * width + j, :) = image(i, j, :);
        end
    end
    
    %% calculate kernel matrix
    gammaS = 0.001;
    gammaC = 0.001;
    gram = calKernel(data, width, gammaS, gammaC);
    
    %% kernel k-means algorithm
    for k = 2:4
        output = [filename(1:last) '_kernelkmeans_kmeansplusplus_' num2str(k) '.gif'];
        [curr, count] = kernelKmeans(data, gram, k, height, width, output);
    end
    %% unnormalized spectral clustering
    for k = 2:4
        output = [prefix '_unnormalized_kmeansplusplus_' num2str(k) '.gif'];
        [clusters, count, means] = unnormalizedSpectralClustering(gram, k, height, width, output);
    end
    %% normalized spectral clustering
    for k = 2:4
        output = [prefix '_normalized_kmeansplusplus_' num2str(k) '.gif'];
        [clusters, count, means] = normalizedSpectralClustering(gram, k, height, width, output);
    end
end