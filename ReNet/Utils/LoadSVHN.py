from scipy.io import loadmat

import numpy as np

from Utils.ReduceImbalance import *


def load_mat_file(filename):
    mat_file = loadmat(filename)

    X = mat_file['X']
    y = mat_file['y']

    return X, y

def load_SVHN(path):
    x_train, y_train = load_mat_file(path+'train_32x32.mat')
    x_extra, y_extra = load_mat_file(path+'extra_32x32.mat')

    #x_train = np.load(path+'x_train.npy')
    #y_train = np.load(path+'y_train.npy')
    #x_extra = np.load(path+'x_extra.npy')
    #y_extra = np.load(path+'y_extra.npy')

    x_train = np.moveaxis(x_train, -1, 0)
    x_extra = np.moveaxis(x_extra, -1, 0)

    y_train = np.squeeze(y_train)
    y_extra = np.squeeze(y_extra)

    x_concat = np.vstack([x_train, x_extra])
    y_concat = np.hstack([y_train, y_extra])

    del x_train
    del y_train
    del x_extra
    del y_extra

    x_train, y_train = reduce_imbalance(x_concat, y_concat,
            samples_per_class=54395,
            num_classes=10,
            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    del x_concat
    del y_concat

    x_train = x_train[:543949]
    y_train = y_train[:543949]

    x_test, y_test = load_mat_file(path+'test_32x32.mat')
    #x_test = np.load(path+'x_test.npy')
    #y_test = np.load(path+'y_test.npy')
    x_test = np.moveaxis(x_test, -1, 0)
    y_test = np.squeeze(y_test)

    return x_train, y_train, x_test, y_test
