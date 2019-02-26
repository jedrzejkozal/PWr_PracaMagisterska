from keras import Model
from keras.layers import Dense, Flatten

from HilbertLayer import *
from ModifiedReNetLayer import *


class ModifiedReNet(Model):

    def __init__(self, patch_size, reNet_hidden_size, fully_conn_hidden_size, num_classes):
        super().__init__()

        self.hilbert_layer = HilbertLayer()
        self.reNet = ModifiedReNetLayer(patch_size, reNet_hidden_size)

        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        flat_imgs = self.hilbert_layer(inputs)
        reNet_output = self.reNet(flat_imgs)

        x = self.flatten(reNet_output)
        x = self.dense(x)
        x = self.softmax(x)

        return x
