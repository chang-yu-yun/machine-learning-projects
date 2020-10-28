import numpy as np

def readData(filename, num):
    data = np.loadtxt(filename, dtype=float, delimiter=',')
    data = np.reshape(data, (num, -1))
    return data

def readLabel(filename):
    labels = np.loadtxt(filename, dtype=int)
    return labels
