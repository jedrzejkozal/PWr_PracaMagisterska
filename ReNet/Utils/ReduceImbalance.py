import numpy as np


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
        indexes_chosen.append(np.random.choice(indexes_all[i], samples_per_class))

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
