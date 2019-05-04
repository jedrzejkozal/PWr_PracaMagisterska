from keras.models import Sequential, Model
from keras.optimizers import Adam

from Models.HilbertLayer.HilbertLayer import *

import numpy as np


def test_4x4():
    x = np.array(list(range(16)))
    x = x.reshape((1, 4, 4, 1))

    expected = np.array([12, 13, 9, 8,
                        4, 0, 1, 5,
                        6, 2, 3, 7,
                        11, 10, 14, 15])
    expected = expected.reshape((1, 16, 1))

    model = Sequential()
    model.add(HilbertLayer())

    model.compile(loss='mse',
            optimizer=Adam(),
            metrics=['categorical_accuracy'])
    model.fit(x,
            expected,
            epochs=1,
            batch_size=1)

    result = model.predict(x)
    assert np.isclose(expected, result).all()


def test_8x8():
    x = np.array(list(range(64)))
    x = x.reshape((1, 8, 8, 1))

    expected = np.array([56, 48, 49, 57, 58, 59, 51, 50,
                    42, 43, 35, 34, 33, 41, 40, 32,
                    24, 25, 17, 16, 8, 0, 1, 9,
                    10, 2, 3, 11, 19, 18, 26, 27,
                    28, 29, 21, 20, 12, 4, 5, 13,
                    14, 6, 7, 15, 23, 22, 30, 31,
                    39, 47, 46, 38, 37, 36, 44, 45,
                    53, 52, 60, 61, 62, 54, 55, 63])
    expected = expected.reshape((1, 64, 1))

    model = Sequential()
    model.add(HilbertLayer())

    model.compile(loss='mse',
            optimizer=Adam(),
            metrics=['categorical_accuracy'])
    model.fit(x,
            expected,
            epochs=1,
            batch_size=1)

    result = model.predict(x)
    assert np.isclose(expected, result).all()
