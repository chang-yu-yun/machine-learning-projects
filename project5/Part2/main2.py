import numpy as np
import matplotlib.pyplot as plt
from svmutil import *
from utility import *

if __name__ == '__main__':
    # read the training data
    trainY = readLabel('Y_train.csv')
    numOfData = trainY.size
    trainX = readData('X_train.csv', numOfData)

    # SVM
    """
    svm_train options:
        -s svm_type : set type of SVM (default 0)
	    0 -- C-SVC		(multi-class classification)
        -t kernel_type : set type of kernel function (default 2)
	    2 -- radial basis function: exp(-gamma*|u-v|^2)
        -d degree : set degree in kernel function (default 3)
        -g gamma : set gamma in kernel function (default 1/num_features)
        -r coef0 : set coef0 in kernel function (default 0)
        -c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
        -v n: n-fold cross validation mode
        -q : quiet mode (no outputs)
    """
    logC = np.arange(-4, 5)  # the log value of the parameter C
    logG = np.arange(-4, 5)  # the log value of the parameter gamma
    accuracy = np.empty((logC.size, logG.size))
    
    # grid search   
    for i in logC:
        for j in logG:
            param = '-q -v 3 -c {} -g {}'.format(np.power(2.0, i), np.power(2.0, j))
            accuracy[i][j] = svm_train(trainY, trainX, param)
    
    # store the results to the csv file
    logC = np.arange(-5, 5).reshape(1, -1)
    result = np.vstack((logC, np.hstack((logG.reshape(-1, 1), accuracy))))
    np.savetxt('result.csv', result, fmt='%.2f', delimiter=',')
