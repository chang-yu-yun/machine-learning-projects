import numpy as np
import matplotlib.pyplot as plt
from utility import *

if __name__ == '__main__':
    # initialization
    beta = 5.0
    var = 1.0 / beta
    alpha = 1.0
    scale = 1.0
    x, y = readData()

    # calculate the covariance matrix C
    cov = calCov(x, alpha, scale, var)

    # get the distribution of p(f|x)
    start = -60
    end = 60
    step = 0.01
    xData = np.arange(start, end, step)
    size = xData.size
    fMean = np.empty(size)
    fVar = np.empty(size)
    for i in range(size):
        fMean[i], fVar[i] = getDistribution(xData[i], x, y, cov, var, alpha, scale)

    # get the 95% confidence interval [mean - 1.96 * std, mean + 1.96 * std]
    std = np.sqrt(fVar)  # get the standard deviation, which is the square root of the variance
    upperBound = fMean + 1.96 * std
    lowerBound = fMean - 1.96 * std
    ylimU, ylimL = np.ceil(max(upperBound)), np.ceil(min(lowerBound) - 1)

    # visualize the result
    plt.plot(x, y, 'ro', markersize=8)
    plt.plot(xData, fMean, 'b-', linewidth=2)
    plt.fill_between(xData, lowerBound, upperBound, facecolor='yellow')
    plt.grid(which='both')
    plt.xlim((start, end))
    plt.ylim((ylimL, ylimU))
    plt.show()
