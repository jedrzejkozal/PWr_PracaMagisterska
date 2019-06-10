import pytest
import time
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from TimerCallback import *


@pytest.fixture
def simple_data():
    x = np.random.random((5000, 100))
    y = np.random.random((5000, 2))

    return x, y


@pytest.fixture
def simple_net():
    model = Sequential()
    model.add(Dense(100, activation='tanh'))
    model.add(Dense(2, activation='tanh'))

    model.compile(optimizer='SGD', loss='mse')
    return model


def measure_execution_time_10_times(f):
    def timeit(*args, **kwargs):
        duration = 0.0
        for i in range(10):
            start_time = time.time()
            f(*args, **kwargs)
            duration += time.time() - start_time
        duration = duration / 10
        return duration

    return timeit


@measure_execution_time_10_times
def train_network(model, x, y, epochs=None, callbacks=None):
    model.fit(x, y,
            batch_size=1,
            epochs=epochs,
            callbacks=callbacks)


def test_10_epochs_simple_nets_sum_of_epochs_duration_is_equal_to_time_of_whole_traning(simple_data, simple_net):
    x, y = simple_data
    sut = TimerCallback()

    trainig_duration = train_network(simple_net, x, y, epochs=10, callbacks=[sut])

    results = sut.get_results()
    epochs_sum = np.sum(results, axis=0)
    np.isclose(trainig_duration, epochs_sum, atol=0.5)
