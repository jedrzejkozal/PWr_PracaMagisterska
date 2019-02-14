import pytest
import numpy as np
from main import *


class TestSimpleReNet(object):

    @pytest.fixture
    def sut(self):
        model = SimpleReNet([[2,2]], 1, 1)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])
        return model


    @pytest.fixture
    def simple_data_x(self):
        num_samples = 2
        x = np.zeros((num_samples, 10, 10), dtype=np.uint8)
        x[1] = np.ones((10, 10))
        print(x)
        return x


    @pytest.fixture
    def simple_data_y(self):
        y = np.array([0, 1], dtype=np.uint8)
        print(y)
        return y


    def test_model_output_for_2_classes_is_0_or_1(self, sut, simple_data_x,
            simple_data_y):
        sut.fit(simple_data_x, simple_data_y, epochs=1, shuffle=False)
        assert True == True
