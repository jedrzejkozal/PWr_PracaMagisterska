from keras import Model

from Models.ReNetLayer import *


class MnistReproduction(Model):

    def __init__(self):
        super().__init__()

        self.first_reNetLayer = ReNetLayer([[2, 2]], 128, #256
                use_dropout=True, dropout_rate=0.2,
                is_first_layer=True)
        self.second_reNetLayer = ReNetLayer([[2, 2]], 128, #256
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
        first_reNet_output = self.first_reNetLayer(inputs)
        second_reNet_output = self.second_reNetLayer(first_reNet_output)

        x = self.flatten(second_reNet_output)
        x = self.first_dense(x)
        x = self.first_dropout(x)
        x = self.second_dense(x)
        x = self.second_dropout(x)
        x = self.softmax(x)

        return x
