import pytest
import numpy as np

from Utils.InputNormalization import *


def test_normalize_uniform():
    arg = np.random.random((10,5,5,3))*8 + 10
    assert np.mean(arg) != 0
    assert np.std(arg) != 1

    result = normalize(arg)

    assert np.isclose(np.mean(result), 0)
    assert np.isclose(np.std(result), 1)
