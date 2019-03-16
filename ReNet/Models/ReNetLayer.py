import tensorflow as tf
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, Permute, concatenate
from keras.layers import Layer, Dropout, Masking
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


class ReNetLayer(Layer):

    def __init__(self, size_of_patches, hidden_size,
                    use_dropout=False, dropout_rate=None,
                    is_first_layer=False, input_dim=None):
        super().__init__()

        self.input_dim = input_dim
        self.size_of_patches = size_of_patches
        self.w_p = size_of_patches[0][0]
        self.h_p = size_of_patches[0][1]

        self.is_first_layer = is_first_layer
        if is_first_layer:
            self.mask = Masking(mask_value=float('Inf'))

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

        self.layer_vertical_activations_permutarion = Permute((1, 3, 2))
        self.layer_horizontal_activations_permutarion = Permute((3, 1, 2))


    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super().build(input_shape)  # Be sure to call this at the end


    def compute_output_shape(self, input_shape):
        I = int(input_shape[1]) // self.w_p
        J = int(input_shape[2]) // self.h_p
        return (input_shape[0], I, J, 2*self.hidden_size)


    def call(self, inputs):
        self.__validate_patch_size(inputs.shape)
        self.__initialize_input_size_dependent_variables(inputs.shape)

        vertical_sweep_output = self.vertical_sweep(inputs)
        horizontal_sweep_output = self.horizontal_sweep(vertical_sweep_output)

        return horizontal_sweep_output


    def __validate_patch_size(self, inputs_shape):
        if inputs_shape[2] % self.h_p != 0 or inputs_shape[1] % self.w_p != 0:
            raise ValueError(
            "invalid patches size ({}) "
            "for current image size ({}). "
            "Please resizie image ".format(self.size_of_patches, inputs_shape[1:]))


    def __initialize_input_size_dependent_variables(self, inputs_shape):
        self.I = int(inputs_shape[1]) // self.w_p
        self.J = int(inputs_shape[2]) // self.h_p

        self.layer_vertical_patches_reshape = Reshape((self.J, self.w_p * self.h_p * int(inputs_shape[3])))
        self.layer_horizontal_patches_permute = Permute((2, 3, 1))
        self.layer_precise_tensor_shape = Reshape((self.J, self.I, int(2*self.hidden_size)))


    def vertical_sweep(self, inputs):
        LSTM_outputs = []

        for col in self.get_columns(inputs):
            column_activations = self.get_activations_for_columns(col)
            LSTM_outputs.append(column_activations)

        merged = concatenate(LSTM_outputs, axis=2)
        print("vertical_sweep: merged: ", merged.shape)
        result = self.layer_precise_tensor_shape(merged)
        print("vertical_sweep: result: ", result)
        return result


    def get_columns(self, inputs):
        for j in range(0, inputs.shape[2], self.w_p):
            yield inputs[:, :, j:j+self.w_p, :]


    def get_activations_for_columns(self, col):
        self.get_vertical_patches(col)

        if self.is_first_layer:
            self.patches = self.mask(self.patches)

        up_down_activation, down_up_activation = self.calc_vertical_LSTM_activations()
        print("up_down_activation: ", up_down_activation.shape)

        merged_tensor = self.merge_opossite_directions_LSTM_activations(up_down_activation, down_up_activation)
        return self.layer_vertical_activations_permutarion(merged_tensor)


    def get_vertical_patches(self, column):
        self.patches = self.layer_vertical_patches_reshape(column)


    def calc_vertical_LSTM_activations(self):
        up_down_activation = self.LSTM_up_down(self.patches)
        down_up_activation = self.LSTM_down_up(tf.reverse(self.patches, [-2]))

        if self.use_dropout:
            up_down_activation = self.dropout_up_down(up_down_activation)
            down_up_activation = self.dropout_down_up(down_up_activation)

        return up_down_activation, down_up_activation


    def merge_opossite_directions_LSTM_activations(self, first_tensor, second_tensor):
        merged_vector = concatenate(
                [tf.keras.backend.expand_dims(first_tensor),
                 tf.keras.backend.expand_dims(second_tensor)], axis=2)
        return merged_vector


    def horizontal_sweep(self, inputs):
        LSTM_outputs = []

        for i, row in enumerate(self.get_rows(inputs)):
            row_activations = self.get_activations_for_row(row)
            LSTM_outputs.append(row_activations)

        merged = concatenate(LSTM_outputs, axis=1)
        return self.layer_precise_tensor_shape(merged)


    def get_rows(self, inputs):
        for i in range(0, inputs.shape[1]):
            yield inputs[:, i:i+1, :]


    def get_activations_for_row(self, row):
        self.get_hor_patches(row)

        left_right_activations, right_left_activations = self.calc_horizontal_LSTM_activations()

        merged_tensor = self.merge_opossite_directions_LSTM_activations(left_right_activations, right_left_activations)
        return self.layer_horizontal_activations_permutarion(merged_tensor)


    def get_hor_patches(self, row):
        self.patches = self.layer_horizontal_patches_permute(row)
        self.patches = tf.squeeze(self.patches, axis=3)


    def calc_horizontal_LSTM_activations(self):
        left_right_activations = self.LSTM_left_right(self.patches)
        right_left_activations = self.LSTM_right_left(tf.reverse(self.patches, [-2]))

        if self.use_dropout:
            left_right_activations = self.dropout_left_right(left_right_activations)
            right_left_activations = self.dropout_right_left(right_left_activations)

        return left_right_activations, right_left_activations
