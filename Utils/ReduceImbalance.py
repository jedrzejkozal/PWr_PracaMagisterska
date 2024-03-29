from keras.utils import to_categorical
import numpy as np


def undersample_to_lowest_cardinality_class(x_data, y_data):
    was_one_hot = False
    if is_one_hot(y_data):
        was_one_hot = True
        y_data = convert_from_one_hot_to_labels(y_data)


    bincount = np.bincount(np.squeeze(y_data))
    x_data, y_data = reduce_imbalance(x_data,
                            y_data,
                            samples_per_class=np.min(bincount),
                            num_classes=len(bincount),
                            labels=list(range(len(bincount))))
    if was_one_hot:
        y_data = to_categorical(y_data)
    return x_data, y_data


def is_one_hot(x):
    return len(np.squeeze(x).shape) != 1

def convert_from_one_hot_to_labels(x):
    return np.argwhere(x == 1)[:, 1]


def reduce_imbalance(x_data, y_data,
        samples_per_class=None,
        num_classes=None,
        labels=None):

    assert samples_per_class is not None
    assert num_classes is not None
    assert labels is not None

    if len(y_data.shape) > 1:
        y_data = y_data.flatten()

    indexes_all = []
    for l in labels:
        indexes_all.append(np.argwhere(y_data == l).flatten())

    indexes_chosen = []
    for i in range(num_classes):
        np.random.seed(0)
        indexes_chosen.append(np.random.choice(indexes_all[i], samples_per_class, replace=False))

    del indexes_all

    choosen_samples_x = []
    choosen_samples_y = []
    for i in range(0, num_classes):
        choosen_samples_x.append(x_data[indexes_chosen[i]])
        choosen_samples_y.append(y_data[indexes_chosen[i]])

    del indexes_chosen

    x_data = np.vstack(choosen_samples_x)
    y_data = np.hstack(choosen_samples_y)

    return x_data, y_data
