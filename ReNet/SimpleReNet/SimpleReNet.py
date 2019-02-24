from keras import Model

from ReNetLayer import *

class SimpleReNet(Model):

    def __init__(self, size_of_patches,
            reNet_hidden_size, fully_conn_hidden_size, num_classes,
            use_dropout=False, reNet_dropout=None, fully_conn_dropout=None):
        super().__init__()

        self.use_dropout = use_dropout
        self.reNetLayer = ReNetLayer(size_of_patches, reNet_hidden_size,
                use_dropout=use_dropout, dropout_rate=reNet_dropout)
        self.flatten = Flatten()
        self.dense = Dense(fully_conn_hidden_size, activation='relu')
        if use_dropout:
            self.dropout = Dropout(fully_conn_dropout)
        self.softmax = Dense(num_classes, activation='softmax')


    def call(self, inputs):
        reNet_output = self.reNetLayer(inputs)

        x = self.flatten(reNet_output)
        x = self.dense(x)
        if self.use_dropout:
            x = self.dropout(x)
        x = self.softmax(x)

        return x
