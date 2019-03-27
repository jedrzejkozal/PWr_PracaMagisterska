import pytest

import numpy as np
from keras.layers import Input

from Models.ReNetLayer import *


class ReNetLayerTest(object):

    @pytest.fixture
    def sut(self):
        self.__class__.setup()
        sut = ReNetLayer([[self.w_p, self.h_p]], self.hidden_size)
        input_shape = (self.num_samples, self.img_width, self.img_height,
                self.number_of_channels)
        sut.compute_output_shape(input_shape)
        return sut


    @pytest.fixture
    def simple_data_x(self):
        self.__class__.setup()
        x = np.zeros((self.num_samples, self.img_width, self.img_height,
                self.number_of_channels), dtype=np.uint8)
        x[self.num_samples // 2:] = np.ones(
                    (self.img_width, self.img_height, self.number_of_channels)
                )
        return x


    @pytest.fixture
    def simple_data_y(self):
        self.__class__.setup()
        y = np.zeros((self.num_samples), dtype=np.uint8)
        y[self.num_samples // 2:] = 1
        return y


    def get_result_shape(self, result):
        print("result_shape: ", result.shape)
        return list(map(lambda x: int(x), result.shape[1:]))


    def test_get_columns_outputs_returns_columns_with_valid_shape(self, sut):
        arg = Input((self.img_height, self.img_width, self.number_of_channels)) #10, 10, 1

        for result in sut.get_columns(arg):
            result_shape = self.get_result_shape(result)
            assert result_shape == [self.img_height, self.w_p, self.number_of_channels] #10, 2, 1


    def test_get_columns_generates_5_columns(self, sut):
        arg = Input((self.img_height, self.img_width, self.number_of_channels)) #10, 10, 1

        number_of_col = 0
        for result in sut.get_columns(arg):
            number_of_col += 1

        assert number_of_col == self.I


    def get_tensor_with_invalid_shape(self):
        if self.img_width % self.w_p == 0:
            width = self.img_width + 1
        else:
            width = self.img_width
        return Input((self.img_height, width, self.number_of_channels)) #10, 11, 1


    def test_call_not_even_number_of_columns_for_patches_exception_raised(self, sut):
        arg = self.get_tensor_with_invalid_shape()

        with pytest.raises(ValueError) as err:
            result = sut.call(arg)
            assert result is None

        assert "invalid patches size" in str(err.value)


    def test_get_vert_patches_returns_patches_with_valid_shape(self, sut, simple_data_x):
        arg = Input((self.img_height, self.w_p, self.number_of_channels))
        sut.J = self.J
        sut.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * self.number_of_channels))

        result = sut.get_vertical_patches(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [self.J, self.h_p*self.w_p*self.number_of_channels]


    def test_vertical_sweep_output_shape_is_J_I_2(self, sut, simple_data_x):
        arg = Input((self.img_height, self.img_width, self.number_of_channels))
        sut.I = self.I
        sut.J = self.J
        sut.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * self.number_of_channels))
        sut.layer_precise_tensor_shape = Reshape((self.J, self.I, int(2*self.hidden_size)))

        result = sut.vertical_sweep(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.hidden_size]


    def test_get_rows_outputs_returns_rows_with_valid_shape(self, sut):
        arg = Input((self.J, self.I, 2*self.hidden_size)) #5, 5, 2

        for result in sut.get_rows(arg):
            result_shape = self.get_result_shape(result)
            assert result_shape == [1, self.I, 2*self.hidden_size]


    def test_get_rows_generates_5_rows(self, sut):
        arg = Input((self.J, self.I, 2*self.hidden_size)) #5, 5, 2

        number_of_col = 0
        for result in sut.get_rows(arg):
            number_of_col += 1

        assert number_of_col == self.J


    def test_get_hor_patches_returns_patches_with_valid_shape(self, sut, simple_data_x):
        arg = Input((1, self.I, 2*self.hidden_size))
        sut.I = self.I
        sut.J = self.J
        sut.layer_horizontal_patches_permute = Permute((2, 3, 1))

        result = sut.get_hor_patches(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [self.I, 2*self.hidden_size]



    def test_horizontal_sweep_output_shape_is_J_I_2(self, sut, simple_data_x):
        arg = Input((self.I, self.J, 2*self.hidden_size))
        sut.I = self.I
        sut.J = self.J
        sut.layer_horizontal_patches_permute = Permute((2, 3, 1))
        sut.layer_precise_tensor_shape = Reshape((self.J, self.I, int(2*self.hidden_size)))

        result = sut.horizontal_sweep(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.hidden_size]
