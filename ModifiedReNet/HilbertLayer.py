from keras.layers import Layer, Reshape, concatenate

from HilbertCurve import *

class HilbertLayer(Layer):

    def __init__(self):
        super().__init__()

        self.hilbert_curve = HilbertCurve()


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1]*input_shape[2], input_shape[3])


    def call(self, inputs):
        self.side_length = int(inputs.shape[1])
        self.inputs_shape = inputs.shape[1:3]

        self.reshape = Reshape((1, 1, int(inputs.shape[3])))

        indexes = self.hilbert_curve.get_indexes_vec(self.side_length)

        pixels = []
        for index in indexes:
            pixels.append(self.reshape(inputs[:, index[0, 0], index[0, 1], :]))

        return concatenate(pixels, axis=2)
