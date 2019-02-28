from keras import Model

import sys
sys.path.insert(0, "../")
from ReNetLayer import *

class Cifar10Reproduction(Model):

    def __init__(self):
        super().__init__()

        self.first_reNetLayer = ReNetLayer([[2, 2]], 160) #320)
        self.second_reNetLayer = ReNetLayer([[2, 2]], 160) #320)
        self.third_reNetLayer = ReNetLayer([[2, 2]], 160) #320)
        self.flatten = Flatten()
        fully_conn_hidden_size = 2048 #4096
        self.first_dense = Dense(fully_conn_hidden_size, activation='relu')
        num_classes = 10
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        first_reNet_output = self.first_reNetLayer(inputs)
        second_reNet_output = self.second_reNetLayer(first_reNet_output)
        third_reNet_output = self.third_reNetLayer(second_reNet_output)

        x = self.flatten(third_reNet_output)
        x = self.first_dense(x)
        x = self.softmax(x)

        return x
