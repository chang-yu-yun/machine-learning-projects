import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from utility import *

def objectiveFunc(x, y, var):
    def calVal(theta):
        covY = calCov(y, theta[0], theta[1], var)
        precY = np.linalg.inv(covY)
        size = y.size
        result = 0.5 * (y.T @ precY @ y + np.log(np.linalg.det(covY)) + size * np.log(2 * np.pi))
        return result
    return calVal

if __name__ == '__main__':
    # initialization
    beta = 5.0
    var = 1.0 / beta
    x, y = readData()
    init = [0.01, 0.1, 0, 1, 10]

    # optimize the kernel parameters by minimizing the log marginal likelihood
    alphaBest = init[0]
    scaleBest = init[0]
    bestV = 1000
    for alpha in init:
        for scale in init:
            curr = minimize(objectiveFunc(x, y, var), x0=[alpha, scale], bounds=((1e-5, 1e5), (1e-5, 1e5)))
            if curr.fun < bestV:
                bestV = curr.fun
                alphaBest, scaleBest = curr.x

    print(f'alphaBest: {alphaBest}\nscaleBest: {scaleBest}')

    # calculate the covariance matrix of y
    cov = calCov(x, alphaBest, scaleBest, var)

    # get the distribution of p(f|x)
    start = -60
    end = 60
    step = 0.01
    xData = np.arange(start, end, step)
    size = xData.size
    fMean = np.empty(size)
    fVar = np.empty(size)
    for i in range(size):
        fMean[i], fVar[i] = getDistribution(xData[i], x, y, cov, var, alphaBest, scaleBest)

    # get the 95% confidence interval
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
