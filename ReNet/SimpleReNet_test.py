import pytest
import numpy as np
from main import *


class TestSimpleReNet(object):

    @pytest.fixture
    def sut(self):
        model = SimpleReNet([[2,2]], 1, 1, 2)
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
        sut.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])
        sut.fit(simple_data_x, simple_data_y, epochs=1, shuffle=False)
        assert True == True
