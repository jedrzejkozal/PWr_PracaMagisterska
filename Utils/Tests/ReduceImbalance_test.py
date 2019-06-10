import pytest
import numpy as np

from ReduceImbalance import *


@pytest.fixture
def simple_dataset():
    x_train = np.zeros((50, 10, 10, 3), dtype=np.int8)
    y_train = np.vstack([np.zeros((30,1), dtype=np.int8),
                        np.ones((20,1), dtype=np.int8)]).flatten()
    return x_train, y_train


@pytest.fixture
def one_hot_ecoding_dataset():
    x_train = np.zeros((50, 10, 10, 3), dtype=np.int8)
    y_train = np.zeros((50, 2), dtype=np.int8)
    y_train[:30, 0] = 1
    y_train[30:, 1] = 1
    return x_train, y_train


def test_10_samples_per_class_2_classes_output_shapes_are_valid(simple_dataset):
    x_train, y_train = simple_dataset
    result_x, result_y = reduce_imbalance(x_train, y_train,
            samples_per_class=10,
            num_classes=2,
            labels=[0,1])

    assert result_x.shape == (20, 10, 10, 3)
    assert result_y.shape == (20, )



def test_both_classes_have_equal_number_of_samples(simple_dataset):
    x_train, y_train = simple_dataset
    result_x, result_y = reduce_imbalance(x_train, y_train,
            samples_per_class=10,
            num_classes=2,
            labels=[0,1])

    bincount = np.bincount(result_y.flatten())
    for i in range(1, len(bincount)):
        assert bincount[i] == bincount[i-1]


def test_samples_per_class_param_is_used(simple_dataset):
    x_train, y_train = simple_dataset
    result_x, result_y = reduce_imbalance(x_train, y_train,
            samples_per_class=10,
            num_classes=2,
            labels=[0,1])

    bincount = np.bincount(result_y.flatten())
    for bin in bincount:
        assert bin == 10


def test_not_flattened_dataset(simple_dataset):
    x_train, y_train = simple_dataset
    y_train = y_train.reshape((y_train.shape[0], 1))
    result_x, result_y = reduce_imbalance(x_train, y_train,
            samples_per_class=10,
            num_classes=2,
            labels=[0,1])

    bincount = np.bincount(result_y.flatten())
    for bin in bincount:
        assert bin == 10


def test_to_lowest_simple_datraset_both_classes_have_the_same_num_of_examples(simple_dataset):
    x_train, y_train = simple_dataset
    result_x, result_y = undersample_to_lowest_cardinality_class(x_train, y_train)

    bincount = np.bincount(result_y.flatten())
    assert bincount[0] == bincount[1]


def test_to_lowest_one_hot_econding_both_classes_have_the_same_num_of_examples(one_hot_ecoding_dataset):
    x_train, y_train = one_hot_ecoding_dataset
    result_x, result_y = undersample_to_lowest_cardinality_class(x_train, y_train)

    class_1_examples = np.argwhere(result_y[:, 0] == 1).size
    class_2_examples = np.argwhere(result_y[:, 1] == 1).size
    assert class_1_examples == class_2_examples


def test_one_hot_econding_has_the_same_shape(one_hot_ecoding_dataset):
    x_train, y_train = one_hot_ecoding_dataset
    result_x, result_y = undersample_to_lowest_cardinality_class(x_train, y_train)

    assert result_x.shape == (40, 10, 10, 3)
    assert result_y.shape == (40, 2)


def test_one_hot_econding_every_class_has_20_examples(one_hot_ecoding_dataset):
    x_train, y_train = one_hot_ecoding_dataset
    result_x, result_y = undersample_to_lowest_cardinality_class(x_train, y_train)

    class_1_examples = np.argwhere(result_y[:, 0] == 1).size
    class_2_examples = np.argwhere(result_y[:, 1] == 1).size
    assert class_1_examples == 20
    assert class_2_examples == 20
