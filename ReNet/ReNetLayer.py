import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, Permute, concatenate
from keras.layers import Layer
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K


class ReNetLayer(Layer):


    def __init__(self, size_of_patches, reNet_hidden_size, fully_conn_hidden_size,
                    num_classes):
        self.num_classes = num_classes
        super().__init__()

        self.size_of_patches = size_of_patches
        self.w_p = size_of_patches[0][0]
        self.h_p = size_of_patches[0][1]

        self.reNet_hidden_size = reNet_hidden_size
        self.LSTM_up_down = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_down_up = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_left_right = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_right_left = LSTM(reNet_hidden_size, return_sequences=True)

        self.vertical_single_columns_activations_permutarion = Permute((1, 3, 2))


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


    def get_columns(self, inputs):
        self.__validate_patch_size(inputs)
        for j in range(0, inputs.shape[2], self.w_p):
            yield inputs[:, :, j:j+self.w_p, :]


    def get_hor_patches(self, row):
        permute = Permute((2, 3, 1))
        permuted = permute(row)

        return permuted


    def get_vert_patches(self, column):
        print("__get_patch vec: column:", column)
        reshape = Reshape((self.J, self.w_p * self.h_p * int(column.shape[3])))
        flatten = reshape(column)

        return flatten


    def merge_LSTM_activations(self, activations):
        merged = concatenate([activations[0], activations[1]], axis=2)

        if len(activations) != 2:
            for tensor in activations[2:]:
                merged = concatenate([merged, tensor], axis=2)

        return merged


    def vertical_sweep(self, inputs):
        LSTM_outputs = []

        for col in self.get_columns(inputs):
            print("col: ", col)
            patches = self.get_vert_patches(col)
            print("patches: ", patches)

            up_down_activation = self.LSTM_up_down(patches)
            down_up_activation = self.LSTM_down_up(tf.reverse(patches, [-2]))
            print("up_down_activation: ", up_down_activation)

            merged_vector = concatenate(
                    [tf.keras.backend.expand_dims(up_down_activation),
                     tf.keras.backend.expand_dims(down_up_activation)], axis=2)
            print("merged_vector shape:", merged_vector.shape)
            merged_vector_permuted = self.vertical_single_columns_activations_permutarion(merged_vector)
            print("merged_vector permuted shape:", merged_vector_permuted.shape)

            LSTM_outputs.append(merged_vector_permuted)

            print("\n\n")

        merged = self.merge_LSTM_activations(LSTM_outputs)

        precise_tensor_shape = Reshape((self.J, self.I, int(merged.shape[3])))
        vertical_sweep_output = precise_tensor_shape(merged)

        return vertical_sweep_output


    def horizontal_sweep(self, inputs):
        LSTM_outputs = []

        for row in self.get_rows(inputs):
            patches = self.get_hor_patches(row)
            patches = tf.squeeze(patches, axis=3)

            left_right_activations = self.LSTM_left_right(patches)
            right_left_activations = self.LSTM_right_left(tf.reverse(patches, [-2]))

            merged_vector = concatenate(
                    [tf.keras.backend.expand_dims(left_right_activations),
                     tf.keras.backend.expand_dims(right_left_activations)], axis=2)

            merged_vector_permuted = self.vertical_single_columns_activations_permutarion(merged_vector)
            LSTM_outputs.append(merged_vector_permuted)

        merged = self.merge_LSTM_activations(LSTM_outputs)

        precise_tensor_shape = Reshape((self.J, self.I, int(merged.shape[3])))
        horizontal_sweep_output = precise_tensor_shape(merged)

        return horizontal_sweep_output


    def call(self, inputs):
        print("\n\ninputs: ", inputs)

        self.I = int(inputs.shape[1]) // self.w_p
        self.J = int(inputs.shape[2]) // self.h_p

        vertical_sweep_output = self.vertical_sweep(inputs)
        horizontal_sweep_output = self.horizontal_sweep(vertical_sweep_output)

        return horizontal_sweep_output


    def compute_output_shape(self, input_shape):
        I = int(input_shape[1]) // self.w_p
        J = int(input_shape[2]) // self.h_p
        return (input_shape[0], I, J, 2*self.reNet_hidden_size)
