import tensorflow as tf
from keras.layers import Layer


class InputMaskingLayer(Layer):

    def __init__(self, p):
        super().__init__()

        self.probability = p
        self.mask_value = -100.0
        self.mask_not_drawn = True


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        return input_shape


    def generate_mask(self):
        print("InputMaskingLayer: generate_mask: call")
        random_tensor = tf.keras.backend.random_uniform(self.inputs_shape)
        bool_mask = random_tensor < self.probability
        mask_float = tf.cast(bool_mask, tf.float32)

        inf_tensor = tf.constant(self.mask_value, shape=self.inputs_shape)
        self.mask = tf.multiply(mask_float, inf_tensor)


    def call(self, inputs):
        print("InputMaskingLayer: call: call")

        if self.mask_not_drawn:
            self.inputs_shape = inputs.shape[1:]
            self.mask_not_drawn = False
            self.generate_mask()

        return tf.multiply(inputs, self.mask)
