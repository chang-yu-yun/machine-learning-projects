import os
from util import *
import numpy as np

if __name__ == '__main__':
    path = os.path.join('Yale_Face_Database', 'Training')
    images, labels, height, width = readImages(path)
    path = os.path.join('Yale_Face_Database', 'Testing')
    testImages, testLabels, height, width = readImages(path)
    
    # PCA
    eigenvalues, eigenvectors, mean = PCA(images)
    plotEigenface(eigenvectors, height, width, 'pca')
    projectedImages = eigenvectors.T @ (images - mean)
    recoveredImages = eigenvectors @ projectedImages + mean
    plotReconstruction(images, recoveredImages, height, width, 'pca')
    acc = predict(testImages, testLabels, eigenvectors, projectedImages, labels, mean)
    print('[main] PCA accuracy: {:.2f}%'.format(acc * 100))
    
    # LDA
    eigenvaluesLDA, eigenvectorsLDA = LDA(images, labels)
    plotEigenface(eigenvectorsLDA, height, width, 'lda')
    projectedImagesLDA = eigenvectorsLDA.T @ images
    recoveredImagesLDA = eigenvectorsLDA @ projectedImagesLDA
    plotReconstruction(images, recoveredImagesLDA, height, width, 'lda')
    acc = predict(testImages, testLabels, eigenvectorsLDA, projectedImagesLDA, labels)
    print('[main] LDA accuracy: {:.2f}%'.format(acc * 100))

    # kernel PCA
    kernelTypes = {0 : 'Linear kernel', 1 : 'Polynomial kernel', 2 : 'RBF kernel'}
    for kernelType in range(len(kernelTypes)):
        eigenvectors, kernelMatPCA = kernelPCA(images, kernelType)
        projectedImages = eigenvectors.T @ kernelMatPCA
        testKernelMatPCA = getKernel(images, testImages, kernelType)
        acc = predict(testKernelMatPCA, testLabels, eigenvectors, projectedImages, labels)
        print('[main] kernel PCA with {} accuracy: {:.2f}%'.format(kernelTypes[kernelType], acc * 100))