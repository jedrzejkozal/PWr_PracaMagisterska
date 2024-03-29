import pytest
from keras.layers import Input

from Models.ModifiedTwoLayers import *


class TestModifiedTwoLayers(object):

    @classmethod
    def setup_model_params(cls):
        cls.patch_size = 4
        cls.reNet_hidden_size = 3
        cls.fully_conn_hidden_size = 5


    @classmethod
    def setup(cls):
        cls.num_samples = 10 #60
        cls.img_width = 8
        cls.img_height = 8
        cls.number_of_channels = 3
        cls.num_classes = 2
        cls.setup_model_params()


    @pytest.fixture
    def sut(self):
        self.__class__.setup()
        model = ModifiedTwoLayers(self.patch_size, self.reNet_hidden_size,
                self.fully_conn_hidden_size, self.num_classes)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])
        return model


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


    def test_model_output_for_2_classes_shape_is_num_samples_x_2(self, sut,
            simple_data_x, simple_data_y):
        sut.fit(simple_data_x, simple_data_y, epochs=1, shuffle=False)
        result = sut.predict(simple_data_x)
        assert result.shape == (self.num_samples, 2)


    def test_call_returns_tensor_with_valid_shape(self, sut, simple_data_x,
            simple_data_y):
        arg = Input((self.img_height, self.img_width, self.number_of_channels))

        result = sut.call(arg)
        result_shape = self.get_result_shape(result)

        assert result_shape == [2]
