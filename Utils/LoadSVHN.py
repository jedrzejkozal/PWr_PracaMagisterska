from scipy.io import loadmat
from os.path import join

import numpy as np

from Utils.ReduceImbalance import *


def load_mat_file(filename):
    mat_file = loadmat(filename)

    X = mat_file['X']
    y = mat_file['y']

    return X, y

def load_SVHN(path):
    x_train = np.load(join(path, 'x_train.npy'), allow_pickle=True, fix_imports=False)
    y_train = np.load(join(path, 'y_train.npy'), allow_pickle=True, fix_imports=False)
    x_extra = np.load(join(path, 'x_extra.npy'), allow_pickle=True, fix_imports=False)
    y_extra = np.load(join(path, 'y_extra.npy'), allow_pickle=True, fix_imports=False)

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

    #x_train, y_train = reduce_imbalance(x_concat, y_concat,
    #        samples_per_class=54395,
    #        num_classes=10,
    #        labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    #del x_concat
    #del y_concat

    x_train = x_concat[:543949]
    y_train = y_concat[:543949]

    del x_concat
    del y_concat

    x_test = np.load(join(path, 'x_test.npy'), allow_pickle=True, fix_imports=False)
    y_test = np.load(join(path, 'y_test.npy'), allow_pickle=True, fix_imports=False)
    x_test = np.moveaxis(x_test, -1, 0)
    y_test = np.squeeze(y_test)

    y_train = y_train - 1
    y_test = y_test - 1

    return x_train, y_train, x_test, y_test
