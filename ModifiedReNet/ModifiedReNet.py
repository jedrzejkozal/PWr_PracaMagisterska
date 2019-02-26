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
        print("\n\ninputs: ", inputs)
        flat_imgs = self.hilbert_layer(inputs)
        print("flatted: ", flat_imgs.shape)
        reNet_output = self.reNet(flat_imgs)
        print("reNet_output: ", reNet_output.shape)

        x = self.flatten(reNet_output)
        print("flatten: ", x.shape)
        #x = self.dense(x)
        #print("dense: ", x.shape)
        #x = self.softmax(x)
        #print("softmax: ", x.shape)

        return x
