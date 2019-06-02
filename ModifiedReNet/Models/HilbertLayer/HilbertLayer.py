from keras.layers import Layer, Reshape, concatenate

from ModifiedReNet.Models.HilbertLayer.HilbertCurve import *

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
        self.__initialize_input_size_dependent_variables(inputs.shape)
        vec = self.__convert_to_1D(inputs)

        return concatenate(vec, axis=2)


    def __initialize_input_size_dependent_variables(self, inputs_shape):
        self.side_length = int(inputs_shape[1])
        self.all_channels_reshape = Reshape((1, 1, int(inputs_shape[3])))


    def __convert_to_1D(self, inputs):
        pixels = []
        for index in self.__get_indexes():
            pixels.append(self.__get_pixel_at_index(inputs, index))
        return pixels


    def __get_indexes(self):
        return self.hilbert_curve.get_indexes_vec(self.side_length)


    def __get_pixel_at_index(self, inputs, index):
        return self.all_channels_reshape(inputs[:, index[0, 0], index[0, 1], :])
