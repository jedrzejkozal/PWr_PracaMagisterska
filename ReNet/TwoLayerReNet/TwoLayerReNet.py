from ReNetLayer import *

class TwoLayerReNet(Model):

    def __init__(self, size_of_patches, reNet_hidden_size, fully_conn_hidden_size, num_classes):
        super().__init__()

        self.firstReNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size, fully_conn_hidden_size, num_classes)
        self.secondReNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size, fully_conn_hidden_size, num_classes)
        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        first_ReNet_output = self.firstReNetLayer(inputs)
        second_ReNet_output = self.secondReNetLayer(first_ReNet_output)

        x = self.flatten(second_ReNet_output)
        x = self.dense(x)
        x = self.softmax(x)

        return x
