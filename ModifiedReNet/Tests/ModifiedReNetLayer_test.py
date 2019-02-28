import pytest
from keras.layers import Input

from Models.ModifiedReNetLayer import *


class TestModifiedReNetLayer(object):

    @pytest.fixture
    def sut(self):
        self.patch_size = 4
        self.hidden_size = 3
        return ModifiedReNetLayer(self.patch_size, self.hidden_size)


    def get_result_shape(self, result):
        return list(map(lambda x: int(x), result.shape[1:]))


    def test_compute_output_shape_returns_valid_shape_for_input_8x8x3(self, sut):
        arg = (1, 64, 3)

        result = sut.compute_output_shape(arg)
        assert result == (1, 16, 6)


    def test_call_for_64x3_returns_validsize_tensor(self, sut):
        arg = Input((1, 64, 3))

        result = sut.call(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [1, 16, 2*self.hidden_size]
