import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, Permute, concatenate
from keras.layers import Layer, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


class ReNetLayer(Layer):

    def __init__(self, size_of_patches, hidden_size,
                    use_dropout=False, dropout_rate=None):
        super().__init__()

        self.size_of_patches = size_of_patches
        self.w_p = size_of_patches[0][0]
        self.h_p = size_of_patches[0][1]

        self.hidden_size = hidden_size
        self.LSTM_up_down = LSTM(hidden_size, return_sequences=True)
        self.LSTM_down_up = LSTM(hidden_size, return_sequences=True)
        self.LSTM_left_right = LSTM(hidden_size, return_sequences=True)
        self.LSTM_right_left = LSTM(hidden_size, return_sequences=True)

        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout_up_down = Dropout(dropout_rate)
            self.dropout_down_up = Dropout(dropout_rate)
            self.dropout_left_right = Dropout(dropout_rate)
            self.dropout_right_left = Dropout(dropout_rate)

        self.vert_LSTM_activations_permutarion = Permute((1, 3, 2))
        self.hor_LSTM_activations_permutarion = Permute((3, 1, 2))


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def __validate_patch_size(self, inputs):
        if inputs.shape[2] % self.h_p != 0 or inputs.shape[1] % self.w_p != 0:
            raise ValueError(
                "invalid patches size ({}) "
                "for current image size ({}). "
                "Please resizie image ".format(self.size_of_patches, inputs.shape[1:]))


    def get_rows(self, inputs):
        for i in range(0, inputs.shape[1]):
            yield inputs[:, i:i+1, :]
        self.vert_LSTM_activations_permutarion = Permute((1, 3, 2))


    def get_columns(self, inputs):
        for j in range(0, inputs.shape[2], self.w_p):
            yield inputs[:, :, j:j+self.w_p, :]


    def get_hor_patches(self, row):
        self.patches = self.hor_patch_permute(row)
        self.patches = tf.squeeze(self.patches, axis=3)


    def get_vert_patches(self, column):
        self.patches = self.vert_patches_reshape(column)


    def merge_single_LSTM_activations(self, first_tensor, second_tensor):
        merged_vector = concatenate(
                [tf.keras.backend.expand_dims(first_tensor),
                 tf.keras.backend.expand_dims(second_tensor)], axis=2)
        return merged_vector


    def vertical_sweep(self, inputs):
        LSTM_outputs = []

        for col in self.get_columns(inputs):
            self.get_vert_patches(col)

            up_down_activation = self.LSTM_up_down(self.patches)
            down_up_activation = self.LSTM_down_up(tf.reverse(self.patches, [-2]))

            if self.use_dropout:
                up_down_activation = self.dropout_up_down(up_down_activation)
                down_up_activation = self.dropout_down_up(down_up_activation)

            merged_tensor = self.merge_single_LSTM_activations(up_down_activation, down_up_activation)
            merged_tensor = self.vert_LSTM_activations_permutarion(merged_tensor)
            LSTM_outputs.append(merged_tensor)

        merged = concatenate(LSTM_outputs, axis=2)
        vertical_sweep_output = self.precise_tensor_shape(merged)

        return vertical_sweep_output


    def horizontal_sweep(self, inputs):
        LSTM_outputs = []

        for i, row in enumerate(self.get_rows(inputs)):
            self.get_hor_patches(row)

            left_right_activations = self.LSTM_left_right(self.patches)
            right_left_activations = self.LSTM_right_left(tf.reverse(self.patches, [-2]))

            if self.use_dropout:
                left_right_activations = self.dropout_left_right(left_right_activations)
                right_left_activations = self.dropout_right_left(right_left_activations)

            merged_tensor = self.merge_single_LSTM_activations(left_right_activations, right_left_activations)
            merged_tensor = self.hor_LSTM_activations_permutarion(merged_tensor)
            LSTM_outputs.append(merged_tensor)

        merged = concatenate(LSTM_outputs, axis=1)
        horizontal_sweep_output = self.precise_tensor_shape(merged)

        return horizontal_sweep_output


    def call(self, inputs):
        self.__validate_patch_size(inputs)

        self.I = int(inputs.shape[1]) // self.w_p
        self.J = int(inputs.shape[2]) // self.h_p

        self.vert_patches_reshape = Reshape((self.J, self.w_p * self.h_p * int(inputs.shape[3])))
        self.hor_patch_permute = Permute((2, 3, 1))
        self.precise_tensor_shape = Reshape((self.J, self.I, int(2*self.hidden_size)))

        vertical_sweep_output = self.vertical_sweep(inputs)
        horizontal_sweep_output = self.horizontal_sweep(vertical_sweep_output)

        return horizontal_sweep_output


    def compute_output_shape(self, input_shape):
        I = int(input_shape[1]) // self.w_p
        J = int(input_shape[2]) // self.h_p
        return (input_shape[0], I, J, 2*self.hidden_size)
