import numpy as np

def readData():
    x, y = np.loadtxt('input.data', dtype=[('col0', 'f8'), ('col1', 'f8')], unpack=True)
    return x, y

def calKernel(d1, d2, alpha, scale):
    diff = np.absolute(d1 - d2)
    value = np.power(1.0 + np.power(diff, 2.0) / (2.0 * alpha * np.power(scale, 2.0)), -alpha)
    return value

def calCov(d, alpha, scale, var):
    size = d.size
    cov = np.empty((size, size))
    for i in range(size):
        for j in range(size):
            cov[i][j] = calKernel(d[i], d[j], alpha, scale)
            if i == j:
                cov[i][j] = cov[i][j] + var
    return cov

def getDistribution(d, x, y, cov, var, alpha, scale):
    size = x.size
    kernelV = np.empty(size)
    for i in range(size):
        kernelV[i] = calKernel(d, x[i], alpha, scale)
    prec = np.linalg.inv(cov)
    variance = calKernel(d, d, alpha, scale) - kernelV.T @ prec @ kernelV
    mean = kernelV.T @ prec @ y
    return mean, variance

