import pytest
from keras.layers import Input

from ModifiedReNetLayer import *


class TestModifiedReNetLayer(object):

    @pytest.fixture
    def sut(self):
        patch_size = 4
        hidden_size = 3
        return ModifiedReNetLayer(patch_size, hidden_size)


    def test_compute_output_shape_returns_valid_shape_for_input_8x8x3(self, sut):
        arg = (None, 8, 8, 3)

        result = sut.compute_output_shape(arg)
        assert result[1:] == (16, 3)
