import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Input, Reshape, concatenate
from keras.preprocessing.sequence import pad_sequences

class SimpleReNet(Model):

    def __init__(self, size_of_patches, reNet_hidden_size, fully_conn_hidden_size):
        super().__init__()

        self.size_of_patches = size_of_patches
        self.w_p = size_of_patches[0][0]
        self.h_p = size_of_patches[0][1]

        self.reNet_hidden_size = reNet_hidden_size
        self.LSTM_up_down = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_down_up = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_left_right = LSTM(reNet_hidden_size, return_sequences=True)
        self.LSTM_right_left = LSTM(reNet_hidden_size, return_sequences=True)

        self.dense = Dense(fully_conn_hidden_size, activation='relu')


    def __get_rows_index_in_input(self, input):
        return 1


    def __get_columns_index_in_input(self, input):
        return 2


    def __get_columns(self, inputs):
        print("__get_columns vec: inputs:", inputs)
        columns_index = self.__get_columns_index_in_input(inputs)
        for i in range(inputs.shape[columns_index]):
            yield i, inputs[:, :, i:i+self.h_p]


    def __get_vert_patches(self, column, column_index):
        print("__get_patch vec: column:", column)
        patches = []
        reshape = Reshape((1, self.w_p * self.h_p))
        for i in range(0, column.shape[2], self.h_p):
            single_patch = tf.slice(column, [0, i, 0], [2, self.w_p, self.h_p]) #FIX ME: hardcoded 2 for now
            flated_slice = reshape(single_patch)
            patches.append(flated_slice)
        return patches

    def call(self, inputs):
        print(inputs)
        print(inputs.shape[1])

        rows_index = self.__get_rows_index_in_input(inputs)
        col_index = self.__get_columns_index_in_input(inputs)
        I = int(inputs.shape[rows_index]) / self.w_p
        J = int(inputs.shape[col_index]) / self.h_p
        horizontal_sweep_input = Input(shape=(J, I)) # 2*self.reNet_hidden_size

        for col_index, col in self.__get_columns(inputs):
            print("col: ", col)
            patches = self.__get_vert_patches(col, col_index)
            print("patches: ", patches)

            up_down_activation = self.LSTM_up_down(patches)
            down_up_activation = self.LSTM_down_up(patches)
            print("up_down_activation: ", up_down_activation)

            merged_vector = concatenate([up_down_activation,
                    down_up_activation], axis=1)
            print("merged_vector shape:", merged_vector.shape)

            horizontal_sweep_input = concatenate([horizontal_sweep_input,
                    merged_vector], axis=1)


        return 1
