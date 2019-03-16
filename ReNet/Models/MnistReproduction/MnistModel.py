from keras.models import Sequential

from Models.ReNetLayer import *
from Models.MnistReproduction.InputMaskingLayer import *


def get_model():
    model = Sequential()
    #model.add(InputMaskingLayer(0.2))

    reNet_hidden_size = 128 #256
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2,
            is_first_layer=False, input_dim=14))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2))

    model.add(Flatten())
    fully_conn_hidden_size = 2048 #4096
    model.add(Dense(fully_conn_hidden_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(fully_conn_hidden_size, activation='relu', name='features'))
    model.add(Dropout(0.2))

    num_classes = 10
    model.add(Dense(num_classes, activation='softmax'))

    return model
