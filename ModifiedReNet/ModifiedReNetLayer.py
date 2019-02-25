from keras.layers import Layer

from HilbertLayer import *

class ModifiedReNetLayer(Layer):

    def __init__(self, patch_size, hidden_size):
        self.patch_size = patch_size
        self.hidden_size = hidden_size


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] * input_shape[2] // self.patch_size,
                self.hidden_size)


    def call(self, inputs):
        pass
