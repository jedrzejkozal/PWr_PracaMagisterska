import numpy as np


def zero_mean(matrix):
    mean = np.mean(matrix, axis=0)
    for i in range(matrix.shape[0]):
        matrix[i] -= mean
    return matrix

def unit_var(matrix):
    std = np.std(matrix, axis=0)
    for i in range(matrix.shape[0]):
        matrix[i] /= std
    return matrix
