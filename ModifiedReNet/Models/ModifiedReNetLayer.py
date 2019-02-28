import tensorflow as tf
from keras.layers import LSTM, Reshape, Permute
from keras.layers import Layer
from keras.layers import concatenate


class ModifiedReNetLayer(Layer):

    def __init__(self, patch_size, hidden_size):
        super().__init__()

        self.patch_size = patch_size
        self.hidden_size = hidden_size

        self.LSTM_forward = LSTM(hidden_size, return_sequences=True)
        self.LSTM_backward = LSTM(hidden_size, return_sequences=True)

        self.output_permutation = Permute((3, 1, 2))


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] // self.patch_size,
                2*self.hidden_size)


    def call(self, inputs):
        self.input_reshape = Reshape((int(inputs.shape[2]) // self.patch_size,
                self.patch_size * int(inputs.shape[3])))
        self.precise_shape = Reshape((int(inputs.shape[2]) // self.patch_size,
                2*self.hidden_size, 1))

        LSTM_input = self.input_reshape(inputs)
        forward_LSTM_output = self.LSTM_forward(LSTM_input)
        backward_LSTM_output = self.LSTM_backward(tf.reverse(LSTM_input, [2]))

        merged = concatenate([forward_LSTM_output, backward_LSTM_output], axis=2)
        merged = self.precise_shape(merged)

        return self.output_permutation(merged)
