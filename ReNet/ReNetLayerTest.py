import pytest

import numpy as np
from keras.layers import Input

from ReNetLayer import *


class ReNetLayerTest(object):

    @pytest.fixture
    def sut(self):
        self.__class__.setup()

        return ReNetLayer([[self.w_p, self.h_p]],
                self.reNet_hidden_size, self.fully_conn_hidden_size, self.num_classes)


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


    def test_get_columns_not_even_number_of_columns_for_patches_exception_raised(self, sut):
        if self.img_width % self.w_p == 0:
            width = self.img_width + 1
        else:
            width = self.img_width
        arg = Input((self.img_height, width, self.number_of_channels)) #10, 11, 1

        with pytest.raises(ValueError) as err:
            for result in sut.get_columns(arg):
                assert result is None

        assert "invalid patches size" in str(err.value)


    def test_get_columns_not_even_number_of_rows_for_patches_exception_raised(self, sut):
        if self.img_height % self.h_p == 0:
            height = self.img_height + 1
        else:
            height = self.img_height
        arg = Input((height, self.img_width, self.number_of_channels)) #11, 10, 1

        with pytest.raises(ValueError) as err:
            for result in sut.get_columns(arg):
                assert result is None

        assert "invalid patches size" in str(err.value)


    def test_get_vert_patches_returns_patches_with_valid_shape(self, sut, simple_data_x):
        arg = Input((self.img_height, self.w_p, self.number_of_channels))
        sut.J = self.J

        result = sut.get_vert_patches(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [self.J, self.h_p*self.w_p*self.number_of_channels]


    def test_merge_LSTM_activations_for_2_tensors_returns_valid_shape(self, sut, simple_data_x):
        arg = [Input((self.J, 1, 2))] * 2

        result = sut.merge_LSTM_activations(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, 2, 2]


    def test_merge_LSTM_activations_for_3_tensors_returns_valid_shape(self, sut, simple_data_x):
        arg = [Input((self.J, 1, 2))] * 3

        result = sut.merge_LSTM_activations(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, 3, 2]


    def test_vertical_sweep_output_shape_is_J_I_2(self, sut, simple_data_x):
        arg = Input((self.img_height, self.img_width, self.number_of_channels))
        sut.I = self.I
        sut.J = self.J

        result = sut.vertical_sweep(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.reNet_hidden_size]


    def test_get_rows_outputs_returns_rows_with_valid_shape(self, sut):
        arg = Input((self.J, self.I, 2*self.reNet_hidden_size)) #5, 5, 2

        for result in sut.get_rows(arg):
            result_shape = self.get_result_shape(result)
            assert result_shape == [1, self.I, 2*self.reNet_hidden_size]


    def test_get_rows_generates_5_rows(self, sut):
        arg = Input((self.J, self.I, 2*self.reNet_hidden_size)) #5, 5, 2

        number_of_col = 0
        for result in sut.get_rows(arg):
            number_of_col += 1

        assert number_of_col == self.J


    def test_get_hor_patches_returns_patches_with_valid_shape(self, sut, simple_data_x):
        arg = Input((1, self.I, 2*self.reNet_hidden_size))
        sut.I = self.I
        sut.J = self.J

        result = sut.get_hor_patches(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [self.I, 2*self.reNet_hidden_size, 1]



    def test_horizontal_sweep_output_shape_is_J_I_2(self, sut, simple_data_x):
        sut.I = self.I
        sut.J = self.J
        arg = Input((self.I, self.J, 2*self.reNet_hidden_size))

        result = sut.horizontal_sweep(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.reNet_hidden_size]
