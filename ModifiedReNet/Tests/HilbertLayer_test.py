import pytest
from keras.layers import Input

from Models.HilbertLayer.HilbertLayer import *


class TestHilbertLayer(object):

    @pytest.fixture
    def sut(self):
        return HilbertLayer()

    def get_result_shape(self, result):
        return list(map(lambda x: int(x), result.shape[1:]))

    def test_compute_output_shape_for_8x8x3_returns_64x3(self, sut):
        arg = (None, 8, 8, 3)

        result = sut.compute_output_shape(arg)
        assert result[1:] == (64, 3)


    def test_call_returns_valid_shape_for_4x4x1_input(self, sut):
        arg = Input((4, 4, 1))

        result = sut.call(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [1, 16, 1]


    def test_call_returns_valid_shape_for_4x4x3_input(self, sut):
        arg = Input((4, 4, 3))

        result = sut.call(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [1, 16, 3]



    def test_call_returns_valid_shape_for_8x8x1_input(self, sut):
        arg = Input((8, 8, 1))

        result = sut.call(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [1, 64, 1]
