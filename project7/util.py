from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
import os, re, math

def readImages(dirPath, shape=(50, 50)):
    fileList = os.listdir(dirPath)
    numOfImages = len(fileList)
    height, width = shape
    images = np.empty((height * width, numOfImages))
    labels = np.empty(numOfImages).astype('uint8')
    for curr, index in zip(fileList, range(numOfImages)):
        labels[index] = int(re.sub(r'\D', '', curr)) - 1
        path = os.path.join(dirPath, curr)
        image = np.asarray(Image.open(path).resize(shape, Image.ANTIALIAS)).flatten()
        images[:, index] = image
    return images, labels, height, width

def PCA(data):
    mean = np.mean(data, axis=1).reshape(-1, 1)
    centeredData = data - mean
    eigenvalues, eigenvectors = np.linalg.eig(centeredData @ centeredData.T) # covariance matrix S = XX'
    sortedIndex = np.argsort(eigenvalues.real)[::-1] # sort the eigenvectors from the largest eigenvalue to the smallest one
    eigenvalues = eigenvalues[sortedIndex]
    eigenvectors = eigenvectors[:, sortedIndex]
    return eigenvalues, eigenvectors, mean

def LDA(data, labels):
    numOfAttr, numOfImages = data.shape
    mean = np.mean(data, axis=1).reshape(-1, 1)
    table = {}
    for i in range(numOfImages):
        table[labels[i]] = 1 if labels[i] not in table else table[labels[i]] + 1
    
    numOfClusters = len(table)
    numOfMembers = table[0]
    clusterMean = np.zeros((numOfAttr, numOfClusters), dtype=np.float64)
    for i in range(numOfImages):
        clusterMean[:, labels[i]] += data[:, i]
    clusterMean = clusterMean / numOfMembers

    # within-class scatter
    sW = np.zeros((numOfAttr, numOfAttr), dtype=np.float64)
    for i in range(numOfImages):
        temp = data[:, i].reshape(-1, 1) - clusterMean[:, labels[i]].reshape(-1, 1)
        sW += temp @ temp.T
    
    # between-class scatter
    sB = np.zeros((numOfAttr, numOfAttr), dtype=np.float64)
    for i in range(numOfClusters):
        temp = clusterMean[:, i].reshape(-1, 1) - mean
        sB += numOfMembers * (temp @ temp.T)

    # get the eigenvectors of inv(sW) @ sB as W
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.pinv(sW, hermitian=True) @ sB)
    sortedIndex = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[sortedIndex]
    eigenvectors = eigenvectors[:, sortedIndex]
    return eigenvalues, eigenvectors[:, :25]

def getKernel(x, y, kernelType=0):
    if kernelType == 1:
        print('[getKernel] Use polynomial kernel with power = 2')
        return np.square(x.T @ y)
    elif kernelType == 2:
        print('[getKernel] Use RBF kernel with gamma = 1e-3')
        gamma = 1e-3
        return np.exp((-gamma) * dist.cdist(x.T, y.T))
    print('[getKernel] Use linear kernel')
    return x.T @ y

def kernelPCA(data, kernelType=0):
    kernel = getKernel(data, data, kernelType)
    numOfImages = (data.shape)[1]
    oneN = np.ones((numOfImages, numOfImages)) / numOfImages
    kernelT = kernel - oneN @ kernel - kernel @ oneN + oneN @ kernel @ oneN
    eigenvalues, eigenvectors = np.linalg.eig(kernelT)
    sortedIndex = np.argsort(eigenvalues.real)[::-1]
    eigenvalues = eigenvalues[sortedIndex]
    eigenvectors = eigenvectors[:, sortedIndex]
    return eigenvectors, kernel

def plotEigenface(eigenvectors, height, width, method, num=25):
    numCols = 5
    numRows = math.ceil(num / numCols)
    figure, axes = plt.subplots(numRows, numCols)
    for i in range(num):
        plt.subplot(numRows, numCols, i + 1)
        plt.imshow(eigenvectors[:, i].real.reshape((height, width)), cmap='gray')
    figure.tight_layout(pad=0.5)
    path = os.path.join(method, 'eigenface.png')
    plt.savefig(path)
    plt.show()

def plotReconstruction(originalImages, recoveredImages, height, width, method, num=10):
    numOfImages = (originalImages.shape)[1]
    candidates = np.random.choice(numOfImages, num, replace=False)
    numRows, numCols = 2, num
    figure, axes = plt.subplots(numRows, numCols)
    for i in range(num):
        plt.subplot(numRows, numCols, i + 1)
        plt.imshow(originalImages[:, candidates[i]].real.reshape((height, width)), cmap='gray')
        plt.subplot(numRows, numCols, numCols + i + 1)
        plt.imshow(recoveredImages[:, candidates[i]].real.reshape((height, width)), cmap='gray')
    figure.tight_layout(pad=0.5)
    plt.savefig(os.path.join(method, 'face_reconstruction.png'))
    plt.show()

def predict(testImages, testLabels, eigenvectors, projectedImages, labels, mean=None, k=3):
    numOfAttr, numOfTestImages = testImages.shape
    numOfImages = (projectedImages.shape)[1]
    if mean is None:
        mean = np.zeros((numOfAttr, 1))
    
    correct = 0
    projectedTestImages = eigenvectors.T @ (testImages - mean)
    # k-NN
    for i in range(numOfTestImages):
        distances = np.empty(numOfImages)
        for j in range(numOfImages):
            distances[j] = np.sum(np.square(projectedTestImages[:, i] - projectedImages[:, j]))
        sortedIndex = np.argsort(distances)
        candidates = labels[sortedIndex][:k]
        label, counts = np.unique(candidates, return_counts=True)
        prediction = (sorted(dict(zip(label, counts)).items(), key=lambda x : x[1], reverse=True))[0][0]
        if prediction == testLabels[i]:
            correct += 1
    acc = correct / numOfTestImages
    return acc