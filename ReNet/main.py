import tensorflow as tf
from keras import Model
from keras.layers import LSTM, Dense, Input, concatenate
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
        columns_index = self.__get_columns_index_in_input(inputs)
        for i in range(inputs.shape[columns_index]):
            yield i, inputs[:, :, i]


    def __get_vert_patch(self, vec):
        for i in range(0, vec.shape[1], self.h_p):
            print("__get_patch vec: ", )
            yield vec[:, i:i+patch_size]


    def call(self, input):
        print(input)
        print(input.shape[1])

        rows_index = self.__get_rows_index_in_input(input)
        col_index = self.__get_columns_index_in_input(input)
        I = int(input.shape[rows_index]) / self.w_p
        J = int(input.shape[col_index]) / self.h_p
        horizontal_sweep_input = Input(shape=(I, J, 2*self.reNet_hidden_size))

        for col_index, col in self.__get_columns(input):
            print("col: ", col)
            #for patch_index, patch in tf.map_fn(self.__get_vert_patch, col):

            patch_LSTM_input = pad_sequences(patch)

            up_down_activation = self.LSTM_up_down(patch_LSTM_input)
            down_up_activation = self.LSTM_down_up(patch)

            merged_vector = concatenate([up_down_activation,
                    down_up_activation], axis=2)
            print("merged_vector shape:", merged_vector.shape)


            horizontal_sweep_input = concatenate([horizontal_sweep_input,
                    merged_vector], axis=1)


        return 1
