import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from svmutil import *
from utility import *

def precomputedKernel(x, gamma):
    # format: 1:K(xi,x1) ... L:K(xi,xL)
    linearKernel = x @ x.T
    # radial basis function: exp(-gamma * |u - v|^2)
    # 'sqeuclidean': computes the squared Euclidean distance |u - v|^2
    rbfKernel = np.exp(-gamma * cdist(x, x, 'sqeuclidean'))
    return linearKernel + rbfKernel
    

if __name__ == '__main__':
    # read the training data
    trainY = readLabel('Y_train.csv')
    numOfData = trainY.size
    trainX = readData('X_train.csv', numOfData)

    # read the test data
    testY = readLabel('Y_test.csv')
    numOfData = testY.size
    testX = readData('X_test.csv', numOfData)

    # svm training
    kernelData = precomputedKernel(trainX, 1)
    prob = svm_problem(trainY, kernelData, isKernel=True)
    param = svm_parameter('-t 4 -c 2 -q')
    model = svm_train(prob, param)

    # get the accuracy value
    label, acc, val = svm_predict(testY, testX, model, '-q')
    print('Linear kernel + RBF kernel: accuracy = {:.2f}%'.format(acc[0]))
