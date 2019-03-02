from keras.layers import Layer
from keras.layers import Input
from tensorflow.keras.backend import random_uniform
import tensorflow as tf


class InputMaskingLayer(Layer):

    def __init__(self, p):
        super().__init__()

        self.probability = p
        self.mask_value = float('Inf')


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        return input_shape


    def call(self, inputs):
        mask = random_uniform(inputs.shape[1:])
        bool_mask = mask < self.probability
        mask_float = tf.cast(bool_mask, tf.float32)
        inf_tensor = tf.constant(self.mask_value, shape=inputs.shape[1:])
        final_mask = tf.multiply(mask_float, inf_tensor)


        result = tf.multiply(inputs, final_mask)
        print(result)
        return result


"""
sess = tf.InteractiveSession()
x = tf.constant([True, False], dtype=bool)
print("x: ", x)
x_float = tf.cast(x, tf.float32)
print("x_float: ", x_float)

x_float = tf.Print(x_float, [x_float], message="This is a: ")

x_float.eval()
"""

i = InputMaskingLayer(0.2)
arg = Input((6, 5, 3))
i.call(arg)
