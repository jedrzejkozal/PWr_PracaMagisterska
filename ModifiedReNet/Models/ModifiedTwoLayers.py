from keras import Model
from keras.layers import Dense, Flatten

from Models.HilbertLayer.HilbertLayer import *
from Models.ModifiedReNetLayer import *


class ModifiedTwoLayers(Model):

    def __init__(self, patch_size, reNet_hidden_size, fully_conn_hidden_size, num_classes):
        super().__init__()

        self.hilbert_layer = HilbertLayer()
        self.first_reNet = ModifiedReNetLayer(patch_size, reNet_hidden_size)
        self.second_reNet = ModifiedReNetLayer(patch_size, reNet_hidden_size)

        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        flat_imgs = self.hilbert_layer(inputs)
        first_reNet_output = self.first_reNet(flat_imgs)
        second_reNet_output = self.second_reNet(first_reNet_output)

        x = self.flatten(second_reNet_output)
        x = self.dense(x)
        x = self.softmax(x)

        return x
