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


    def get_result_shape(self, result):
        return list(map(lambda x: x.value, result.shape[1:]))


    def test_get_columns_outputs_returns_columns_with_valid_shape(self, sut):
        arg = Input((self.img_height, self.img_width, self.number_of_channels))

        for result in sut.get_columns(arg):
            result_shape = self.get_result_shape(result)
            assert result_shape == [self.img_height, self.w_p, self.number_of_channels]


    def test_get_columns_generates_5_columns(self, sut):
        arg = Input((self.img_height, self.img_width, self.number_of_channels))

        number_of_col = 0
        for result in sut.get_columns(arg):
            number_of_col += 1

        assert number_of_col == self.I


    def get_tensor_with_invalid_shape(self):
        if self.img_width % self.w_p == 0:
            width = self.img_width + 1
        else:
            width = self.img_width
        return Input((self.img_height, width, self.number_of_channels))


    def test_call_not_even_number_of_columns_for_patches_exception_raised(self, sut):
        arg = self.get_tensor_with_invalid_shape()

        with pytest.raises(ValueError) as err:
            result = sut.call(arg)
            assert result is None

        assert "invalid patches size" in str(err.value)


    def test_get_vert_patches_returns_patches_with_valid_shape(self, sut):
        arg = Input((self.img_height, self.w_p, self.number_of_channels))
        sut.J = self.J
        sut.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * self.number_of_channels))

        result = sut.get_vertical_patches(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [self.J, self.h_p*self.w_p*self.number_of_channels]


    def test_calc_vertical_LSTM_activations_outputs_shape_is_J_x_hidden_units(self, sut):
        num_features = self.w_p * self.h_p * self.number_of_channels
        arg = Input((self.J, num_features))

        up_down_result, down_up_result = sut.calc_vertical_LSTM_activations(arg)
        up_down_result_shape = self.get_result_shape(up_down_result)
        down_up_result_shape = self.get_result_shape(down_up_result)

        assert up_down_result_shape == [None, self.hidden_size]
        assert down_up_result_shape == [None, self.hidden_size]


    def test_vertical_sweep_output_shape_is_J_x_I_x_2(self, sut):
        arg = Input((self.img_height, self.img_width, self.number_of_channels))
        sut.I = self.I
        sut.J = self.J
        sut.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * self.number_of_channels))
        sut.layer_precise_tensor_shape = Reshape((self.J, self.I, int(2*self.hidden_size)))

        result = sut.vertical_sweep(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.hidden_size]


    def test_get_activations_for_column_ouput_shape_is_J_x_1_x_2hidden_size(self, sut):
        column_shape = (self.img_height, self.w_p, self.number_of_channels)
        arg = Input(column_shape)
        sut.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * self.number_of_channels))

        result = sut.get_activations_for_column(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [None, 1, 2*self.hidden_size]


    def test_get_rows_outputs_returns_rows_with_valid_shape(self, sut):
        arg = Input((self.J, self.I, 2*self.hidden_size))

        for result in sut.get_rows(arg):
            result_shape = self.get_result_shape(result)
            assert result_shape == [1, self.I, 2*self.hidden_size]


    def test_get_rows_generates_J_rows(self, sut):
        arg = Input((self.J, self.I, 2*self.hidden_size))

        number_of_col = 0
        for result in sut.get_rows(arg):
            number_of_col += 1

        assert number_of_col == self.J


    def test_get_hor_patches_returns_patches_with_valid_shape(self, sut):
        arg = Input((1, self.I, 2*self.hidden_size))
        sut.I = self.I
        sut.J = self.J

        result = sut.get_hor_patches(arg)
        result_shape = self.get_result_shape(result)
        assert result_shape == [self.I, 2*self.hidden_size]


    def test_calc_horizontal_LSTM_activations_outputs_shape_is_I_x_hidden_units(self, sut):
        num_features = self.w_p * self.h_p * self.number_of_channels
        arg = Input((self.I, num_features))

        up_down_result, down_up_result = sut.calc_horizontal_LSTM_activations(arg)
        up_down_result_shape = self.get_result_shape(up_down_result)
        down_up_result_shape = self.get_result_shape(down_up_result)

        assert up_down_result_shape == [None, self.hidden_size]
        assert down_up_result_shape == [None, self.hidden_size]


    def test_horizontal_sweep_output_shape_is_J_x_I_x_2hidden_size(self, sut):
        arg = Input((self.I, self.J, 2*self.hidden_size))
        sut.I = self.I
        sut.J = self.J
        sut.layer_precise_tensor_shape = Reshape((self.J, self.I, int(2*self.hidden_size)))

        result = sut.horizontal_sweep(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.hidden_size]


    def test_get_activations_for_row_output_shape_is_I_x_1_x_2hidden_size(self, sut):
        row_shape = (1, self.img_width, self.number_of_channels)
        arg = Input(row_shape)

        result = sut.get_activations_for_row(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [1, None, 2*self.hidden_size]


    def test_call_output_shape_is_J_x_I_x_2hidden_size(self, sut):
        arg = Input((self.img_width, self.img_height, self.number_of_channels))

        result = sut.call(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.J, self.I, 2*self.hidden_size]
