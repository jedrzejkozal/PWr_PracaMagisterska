from ReNetLayer import *

class SimpleReNet(Model):

    def __init__(self, size_of_patches, reNet_hidden_size, fully_conn_hidden_size, num_classes):
        super().__init__()

        self.reNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size)
        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        reNet_output = self.reNetLayer(inputs)

        x = self.flatten(reNet_output)
        x = self.dense(x)
        x = self.softmax(x)

        return x
