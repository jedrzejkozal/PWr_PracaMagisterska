from keras.models import Sequential

from Models.ReNetLayer import *
from Models.InputMaskingLayer import *



def get_cifar10_model():
    model = Sequential()

    #model.add(InputMaskingLayer(0.2))

    reNet_hidden_size = 320
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.7,
            is_first_layer=True))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.7))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.7))
    model.add(Flatten())

    fully_conn_hidden_size = 4096
    model.add(Dense(fully_conn_hidden_size, activation='relu'))
    model.add(Dropout(0.7))

    num_classes = 10
    model.add(Dense(num_classes, activation='softmax'))

    return model
