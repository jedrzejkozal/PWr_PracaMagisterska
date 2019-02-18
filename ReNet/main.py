import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, concatenate
from keras.preprocessing.sequence import pad_sequences
from keras.backend import placeholder

class SimpleReNet(Model):

    def __init__(self, size_of_patches, reNet_hidden_size, fully_conn_hidden_size, num_classes):
        super().__init__()

        self.size_of_patches = size_of_patches
        self.w_p = size_of_patches[0][0]
        self.h_p = size_of_patches[0][1]

        self.reNet_hidden_size = reNet_hidden_size
        self.LSTM_up_down = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_down_up = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_left_right = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_right_left = LSTM(reNet_hidden_size, return_sequences=True)

        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        self.softmax = Dense(num_classes, activation='softmax')


    def __validate_patch_size(self, inputs):
        if inputs.shape[2] % self.h_p != 0:
            raise ValueError(
                "invalid patches size ({}) "
                "for current image size ({}). "
                "Please resizie image ".format(self.size_of_patches, inputs.shape[1:]))


    def get_columns(self, inputs):
        self.__validate_patch_size(inputs)
        for i in range(0, inputs.shape[2], self.h_p):
            yield inputs[:, :, i:i+self.h_p, :]


    def get_vert_patches(self, column):
        print("__get_patch vec: column:", column)
        reshape = Reshape((self.J, self.w_p * self.h_p * int(column.shape[3])))

        flatten = reshape(column)

        return flatten


    def call(self, inputs):
        print("inputs: ", inputs)

        self.I = int(inputs.shape[1]) // self.w_p
        self.J = int(inputs.shape[2]) // self.h_p
        vertical_sweep_output = Input(shape=(self.I, self.J, 2*self.reNet_hidden_size))
        #vertical_sweep_output = placeholder()

        for col in self.get_columns(inputs):
            print("col: ", col)
            patches = self.get_vert_patches(col)
            print("patches: ", patches)

            up_down_activation = self.LSTM_up_down(patches)
            down_up_activation = self.LSTM_down_up(patches) #FIXME: down up should have reversed patches
            print("up_down_activation: ", up_down_activation)

            merged_vector = concatenate(
                    [tf.keras.backend.expand_dims(up_down_activation),
                    tf.keras.backend.expand_dims(down_up_activation)], axis=2)
            print("merged_vector shape:", merged_vector.shape)

            #vertical_sweep_output = concatenate([vertical_sweep_output,
            #        merged_vector], axis=1)

            print("\n\n")

        r = Reshape((5, 2, 1))
        merged_vector = r(merged_vector)

        x = self.flatten(merged_vector)
        x = self.dense(x)
        x = self.softmax(x)

        return x
