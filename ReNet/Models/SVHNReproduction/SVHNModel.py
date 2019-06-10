from keras.models import Sequential

from ReNet.Models.ReNetLayer import *


def get_svhn_model():
    model = Sequential()

    reNet_hidden_size = 256
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1,
            is_first_layer=True))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))

    model.add(Flatten())
    fully_conn_hidden_size = 4096
    model.add(Dense(fully_conn_hidden_size, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(fully_conn_hidden_size+1, activation='relu', name='features'))
    model.add(Dropout(0.1))

    num_classes = 10
    model.add(Dense(num_classes, activation='softmax'))

    return model
