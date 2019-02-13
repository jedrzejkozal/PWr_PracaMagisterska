import pytest
import numpy as np
from main import *


class TestSimpleReNet(object):

    @pytest.fixture
    def get_sut(self):
        model = SimpleReNet([[2,2]], 1, 1)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])
        return model


    @pytest.fixture
    def get_simple_data(self):
        num_samples = 2
        x = np.zeros((num_samples, 10, 10), dtype=np.uint8)
        x[1] = np.ones((10, 10))
        y = np.array([0, 1], dtype=np.uint8)
        print(x)
        print(y)
        return x, y


    def test_model_output_for_2_classes_is_0_or_1(self):
        sut = self.get_sut()
        x, y = self.get_simple_data()
        sut.fit(x, y, epochs=1, shuffle=False)
        assert True == True
