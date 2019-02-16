import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, Permute, concatenate
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


    def __check_for_even_number_of_columns(self, inputs):
        if inputs.shape[2] % self.h_p != 0:
            raise ValueError(
                "invalid patches size ({}) "
                "for current image size ({}). "
                "Please resizie image ".format(self.size_of_patches, inputs.shape[1:]))


    def get_columns(self, inputs):
        self.__check_for_even_number_of_columns(inputs)
        for i in range(0, inputs.shape[2], self.h_p):
            yield inputs[:, :, i:i+self.h_p, :]


    def __get_vert_patches(self, column):
        print("__get_patch vec: column:", column)
        reshape = Reshape((self.J, self.w_p * self.h_p * int(column.shape[3])))
        permute = Permute((2, 1))

        flatten = reshape(column)
        patches = permute(flatten)

        return patches


    def call(self, inputs):
        print("inputs: ", inputs)

        self.I = int(int(inputs.shape[1]) / self.w_p)
        self.J = int(int(inputs.shape[2]) / self.h_p)
        #horizontal_sweep_input = Input(shape=(self.I, self.J, 2*self.reNet_hidden_size))
        horizontal_sweep_input = placeholder()

        for col in self.get_columns(inputs):
            print("col: ", col)
            patches = self.__get_vert_patches(col)
            print("patches: ", patches)

            up_down_activation = self.LSTM_up_down(patches)
            #down_up_activation = self.LSTM_down_up(patches)
            print("up_down_activation: ", up_down_activation)

            #merged_vector = concatenate(
            #        [tf.keras.backend.expand_dims(up_down_activation),
            #        tf.keras.backend.expand_dims(down_up_activation)], axis=3) #axis should be 3
            #print("merged_vector shape:", merged_vector.shape)

            #horizontal_sweep_input = concatenate([horizontal_sweep_input,
            #        merged_vector], axis=1)

            print("\n\n")

        x = self.flatten(up_down_activation)
        x = self.dense(x)
        x = self.softmax(x)

        return x
