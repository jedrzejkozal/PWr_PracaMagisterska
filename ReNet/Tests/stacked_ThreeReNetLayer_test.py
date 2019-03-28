import pytest

import numpy as np
from keras.layers import Input
from keras.models import Model

from Models.ReNetLayer import *


class TestStackedThreeReNetLayer(object):

    @classmethod
    def setup_model_params(cls):
        cls.w_p = 2
        cls.h_p = 2
        cls.I = cls.img_width // cls.w_p
        cls.J = cls.img_height // cls.h_p
        cls.hidden_size = 5


    @classmethod
    def setup(cls):
        cls.num_samples = 60
        cls.img_width = 16
        cls.img_height = 16
        cls.number_of_channels = 1
        cls.setup_model_params()

    @pytest.fixture
    def sut(self):
        self.__class__.setup()
        input = Input((self.img_width, self.img_height, self.number_of_channels))
        first_output = ReNetLayer([[self.w_p, self.h_p]], self.hidden_size)(input)
        second_output = ReNetLayer([[self.w_p, self.h_p]], self.hidden_size)(first_output)
        third_output = ReNetLayer([[self.w_p, self.h_p]], self.hidden_size)(second_output)
        sut = Model(inputs=input, outputs=third_output)
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
        return list(map(lambda x: x.value, result.shape[1:]))


    def test_call_output_shape_is_valid(self, sut):
        arg = Input((self.img_width, self.img_height, self.number_of_channels))
        result = sut.call(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [self.img_width // (3*self.w_p),
                self.img_height // (3*self.h_p),
                2*self.hidden_size]
