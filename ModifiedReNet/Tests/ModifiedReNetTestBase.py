import pytest
from keras.layers import Input

from Models.ModifiedReNet import *


class ModifiedReNetTestBase(object):

    @pytest.fixture
    def sut(self):
        self.__class__.setup()
        model = ModifiedReNet(self.patch_size, self.reNet_hidden_size,
                self.fully_conn_hidden_size, self.num_classes,
                self.use_dropout, self.dropout_rate)
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
