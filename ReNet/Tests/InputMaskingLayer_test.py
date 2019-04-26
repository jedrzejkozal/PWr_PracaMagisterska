import pytest
import tensorflow as tf

from Models.InputMaskingLayer import *


class TestInputMaskingLayer(object):

    def test_aaa(self):
        sess = tf.InteractiveSession()
        x = tf.constant([True, False], dtype=bool)
        print("x: ", x)
        x_float = tf.cast(x, tf.float32)
        print("x_float: ", x_float)
        x_float = tf.Print(x_float, [x_float], message="This is a: ")
        x_float.eval()
        assert True


    @pytest.fixture
    def sut(self):
        input_shape = (5, 5, 3)
        inputMaskingLayer = InputMaskingLayer(0.2)
        inputMaskingLayer.build(input_shape)
        return inputMaskingLayer


    def test_generate_mask(self, sut):
        input_shape = (5, 5, 3)
        sut.inputs_shape = input_shape
        sut.generate_mask()
        print(sut.mask)
        assert True
