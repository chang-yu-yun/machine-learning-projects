import numpy as np
import matplotlib.pyplot as plt
from svmutil import *
from utility import *

if __name__ == '__main__':
    # read the training data
    trainY = readLabel('Y_train.csv')
    numOfData = trainY.size
    trainX = readData('X_train.csv', numOfData)

    # read the test data
    testY = readLabel('Y_test.csv')
    numOfData = testY.size
    testX = readData('X_test.csv', numOfData)

    # svm
    kernelType = {'Linear' : '-t 0', 'Polynomial' : '-t 1', 'RBF' : '-t 2'}
    accuracy = {}
    index = 0
    for k, v in kernelType.items():
        model = svm_train(trainY, trainX, '-q ' + v)
        labels, acc, vals = svm_predict(testY, testX, model)
        accuracy[k] = acc[0]

    # print out the results
    for k, v in accuracy.items():
        print('{} kernel: accuracy = {:.2f}%'.format(k, v))
