from scipy.io import loadmat
from os.path import join

import numpy as np


def load_mat_file(filename):
    mat_file = loadmat(filename)

    X = mat_file['X']
    y = mat_file['y']

    return X, y


def save_matrix(matrix, filename):
    np.save(filename, matrix, allow_pickle=False)

x_train, y_train = load_mat_file(join(path,'train_32x32.mat'))
save_matrix(x_train, 'x_train.npy')
save_matrix(y_train, 'y_train.npy')
del x_train
del y_train

x_extra, y_extra = load_mat_file(join(path,'extra_32x32.mat'))
save_matrix(x_extra, 'x_extra.npy')
save_matrix(y_extra, 'y_extra.npy')
del x_extra
del y_extra

x_test, y_test = load_mat_file(join(path,'test_32x32.mat'))
save_matrix(x_test, 'x_test.npy')
save_matrix(y_test, 'y_test.npy')
del x_test
del y_test
