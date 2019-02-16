import pytest
import numpy as np
from keras.layers import Input

from main import *


class TestSimpleReNet(object):

    @pytest.fixture
    def sut(self):
        model = SimpleReNet([[2,2]], 1, 1, 2)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])
        return model


    @pytest.fixture
    def simple_data_x(self):
        num_samples = 60
        x = np.zeros((num_samples, 10, 10, 1), dtype=np.uint8)
        x[30:] = np.ones((10, 10, 1))
        return x


    @pytest.fixture
    def simple_data_y(self):
        y = np.zeros((60), dtype=np.uint8)
        y[30:] = 1
        print(y)
        return y


    def test_model_output_for_2_classes_is_0_or_1(self, sut, simple_data_x,
            simple_data_y):
        sut.fit(simple_data_x, simple_data_y, epochs=1, shuffle=False)
        assert sut.predict(simple_data_x).all() == 0 or sut.predict(simple_data_x).all() == 1


    def test_get_columns_outputs_valid_columns_shape(self, sut):
        arg = Input((10, 10, 1))

        for result in sut.get_columns(arg):
            assert result.shape[1] == 10
            assert result.shape[2] == 2
            assert result.shape[3] == 1


    def test_get_columns_generates_5_columns(self, sut):
        arg = Input((10, 10, 1))

        number_of_col = 0
        for result in sut.get_columns(arg):
            number_of_col += 1

        assert number_of_col == 5 # Y / h_p


    def test_get_columns_not_even_number_of_columns_for_patches_exception_raised(self, sut):
        arg = Input((10, 11, 1))

        with pytest.raises(ValueError) as err:
            for result in sut.get_columns(arg):
                assert result is None

        assert "invalid patches size" in str(err.value)
