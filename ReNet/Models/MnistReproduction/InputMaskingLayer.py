import tensorflow as tf
from keras.layers import Layer


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
        random_tensor = tf.keras.backend.random_uniform(inputs.shape[1:])
        bool_mask = random_tensor < self.probability
        mask_float = tf.cast(bool_mask, tf.float32)

        inf_tensor = tf.constant(self.mask_value, shape=inputs.shape[1:])
        final_mask = tf.multiply(mask_float, inf_tensor)

        return tf.multiply(inputs, final_mask)
