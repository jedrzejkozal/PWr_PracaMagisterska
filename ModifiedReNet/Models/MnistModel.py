from keras import Model
from keras.layers import Dense, Flatten, Dropout

from Models.HilbertLayer.HilbertLayer import *
from Models.ModifiedReNetLayer import *


class MnistModel(Model):

    def __init__(self):
        super().__init__()

        self.hilbert_layer = HilbertLayer()
        self.first_reNetLayer = ModifiedReNetLayer(4, 128)
        self.second_reNetLayer = ModifiedReNetLayer(4, 128)

        self.flatten = Flatten()
        fully_conn_hidden_size = 2048
        self.first_dense = Dense(fully_conn_hidden_size, activation='relu')
        self.first_dropout = Dropout(0.2)
        self.second_dense = Dense(fully_conn_hidden_size, activation='relu')
        self.second_dropout = Dropout(0.2)

        num_classes = 10
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        flat_imgs = self.hilbert_layer(inputs)
        first_reNet_output = self.first_reNetLayer(flat_imgs)
        second_reNet_output = self.second_reNetLayer(first_reNet_output)

        x = self.flatten(second_reNet_output)
        x = self.first_dense(x)
        x = self.first_dropout(x)
        x = self.second_dense(x)
        x = self.second_dropout(x)
        x = self.softmax(x)

        return x
