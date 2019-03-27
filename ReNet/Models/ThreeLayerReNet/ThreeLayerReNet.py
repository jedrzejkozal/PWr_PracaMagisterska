from keras import Model

from Models.ReNetLayer import *

class ThreeLayerReNet(Model):

    def __init__(self, size_of_patches, reNet_hidden_size,
            fully_conn_hidden_size, num_classes):
        super().__init__()

        self.firstReNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size)
        self.secondReNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size)
        self.thirdReNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size)
        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        first_ReNet_output = self.firstReNetLayer(inputs)
        second_ReNet_output = self.secondReNetLayer(first_ReNet_output)
        third_ReNet_output = self.thirdReNetLayer(second_ReNet_output)

        x = self.flatten(third_ReNet_output)
        x = self.dense(x)
        x = self.softmax(x)

        return x
