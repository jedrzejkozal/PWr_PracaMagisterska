from scipy.io import loadmat

import numpy as np


def load_mat_file(filename):
    mat_file = loadmat(filename)

    X = mat_file['X']
    y = mat_file['y']

    return X, y

def load_SVHN(path):
    x_train, y_train = load_mat_file(path+'train_32x32.mat')
    x_test, y_test = load_mat_file(path+'test_32x32.mat')
    x_extra, y_extra = load_mat_file(path+'extra_32x32.mat')

    x_train = np.moveaxis(x_train, -1, 0)
    x_test = np.moveaxis(x_test, -1, 0)
    x_extra = np.moveaxis(x_extra, -1, 0)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_extra = np.squeeze(y_extra)


    print("y_train bincount: ", np.bincount(y_train))
    print("y_test bincount: ", np.bincount(y_test))
    print("y_extra bincount: ", np.bincount(np.squeeze(y_extra)))


    x_concat = np.vstack([x_train, x_extra])
    y_concat = np.hstack([y_train, y_extra])

    del x_train
    del y_train
    del x_extra
    del y_extra

    class_1_all_indexes = np.argwhere(y_concat == 1).flatten()
    class_2_all_indexes = np.argwhere(y_concat == 2).flatten()
    class_3_all_indexes = np.argwhere(y_concat == 3).flatten()
    class_4_all_indexes = np.argwhere(y_concat == 4).flatten()
    class_5_all_indexes = np.argwhere(y_concat == 5).flatten()
    class_6_all_indexes = np.argwhere(y_concat == 6).flatten()
    class_7_all_indexes = np.argwhere(y_concat == 7).flatten()
    class_8_all_indexes = np.argwhere(y_concat == 8).flatten()
    class_9_all_indexes = np.argwhere(y_concat == 9).flatten()
    class_10_all_indexes = np.argwhere(y_concat == 10).flatten()


    class_1_indexes = np.random.choice(class_1_all_indexes, 54395)
    class_2_indexes = np.random.choice(class_2_all_indexes, 54395)
    class_3_indexes = np.random.choice(class_3_all_indexes, 54395)
    class_4_indexes = np.random.choice(class_4_all_indexes, 54395)
    class_5_indexes = np.random.choice(class_5_all_indexes, 54395)
    class_6_indexes = np.random.choice(class_6_all_indexes, 54395)
    class_7_indexes = np.random.choice(class_7_all_indexes, 54395)
    class_8_indexes = np.random.choice(class_8_all_indexes, 54395)
    class_9_indexes = np.random.choice(class_9_all_indexes, 54395)
    class_10_indexes = np.random.choice(class_10_all_indexes, 54395)

    del class_1_all_indexes
    del class_2_all_indexes
    del class_3_all_indexes
    del class_4_all_indexes
    del class_5_all_indexes
    del class_6_all_indexes
    del class_7_all_indexes
    del class_8_all_indexes
    del class_9_all_indexes
    del class_10_all_indexes

    x_train = np.vstack([x_concat[class_1_indexes],
                    x_concat[class_2_indexes],
                    x_concat[class_3_indexes],
                    x_concat[class_4_indexes],
                    x_concat[class_5_indexes],
                    x_concat[class_6_indexes],
                    x_concat[class_7_indexes],
                    x_concat[class_8_indexes],
                    x_concat[class_9_indexes],
                    x_concat[class_10_indexes]
                    ])

    y_train = np.hstack([y_concat[class_1_indexes],
                    y_concat[class_2_indexes],
                    y_concat[class_3_indexes],
                    y_concat[class_4_indexes],
                    y_concat[class_5_indexes],
                    y_concat[class_6_indexes],
                    y_concat[class_7_indexes],
                    y_concat[class_8_indexes],
                    y_concat[class_9_indexes],
                    y_concat[class_10_indexes]
                    ])

    del x_concat
    del y_concat

    del class_1_indexes
    del class_2_indexes
    del class_3_indexes
    del class_4_indexes
    del class_5_indexes
    del class_6_indexes
    del class_7_indexes
    del class_8_indexes
    del class_9_indexes
    del class_10_indexes

    x_train = x_train[:543949]
    y_train = y_train[:543949]


    print("x_train: ", x_train.shape)
    print("y_train: ", y_train.shape)
    print("x_test: ", x_test.shape)
    print("y_test: ", y_test.shape)

    print("y_train bincount: ", np.bincount(np.squeeze(y_train)))
    print("y_test bincount: ", np.bincount(np.squeeze(y_test)))

    return x_train, y_train, x_test, y_test
