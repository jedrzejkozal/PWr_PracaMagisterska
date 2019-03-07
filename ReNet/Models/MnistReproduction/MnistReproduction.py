from keras import Model

from Models.ReNetLayer import *
from Models.MnistReproduction.InputMaskingLayer import *

class MnistReproduction(Model):

    def __init__(self):
        super().__init__()

        self.input_masking = InputMaskingLayer(0.2)

        reNet_hidden_size = 256
        self.first_reNetLayer = ReNetLayer([[2, 2]], reNet_hidden_size,
                use_dropout=True, dropout_rate=0.2,
                is_first_layer=False)
        self.second_reNetLayer = ReNetLayer([[2, 2]], reNet_hidden_size,
                use_dropout=True, dropout_rate=0.2)

        self.flatten = Flatten()
        fully_conn_hidden_size = 2048 #4096
        self.first_dense = Dense(fully_conn_hidden_size, activation='relu')
        self.first_dropout = Dropout(0.2)
        self.second_dense = Dense(fully_conn_hidden_size, activation='relu')
        self.second_dropout = Dropout(0.2)

        num_classes = 10
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        #x = self.input_masking(inputs)

        first_reNet_output = self.first_reNetLayer(inputs)#x)
        second_reNet_output = self.second_reNetLayer(first_reNet_output)

        x = self.flatten(second_reNet_output)
        x = self.first_dense(x)
        x = self.first_dropout(x)
        x = self.second_dense(x)
        x = self.second_dropout(x)
        x = self.softmax(x)

        return x
