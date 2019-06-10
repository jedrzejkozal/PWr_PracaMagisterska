import numpy as np


def normalize(matrix):
    mu = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return (matrix - mu) / std
