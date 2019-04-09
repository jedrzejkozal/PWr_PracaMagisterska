"""
from keras.models import Sequential

from Models.ReNetLayer import *
from Models.InputMaskingLayer import *


def get_svhn_model():
    model = Sequential()
    model.add(InputMaskingLayer(0.2))

    reNet_hidden_size = 5 #256
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2,
            is_first_layer=True))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2))

    model.add(Flatten())
    fully_conn_hidden_size = 20 #4096
    model.add(Dense(fully_conn_hidden_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(fully_conn_hidden_size, activation='relu', name='features'))
    model.add(Dropout(0.2))

    num_classes = 10
    model.add(Dense(num_classes, activation='softmax'))

    return model
"""


from tf.keras import Sequential
from Models.ReNetLayer import *
from Models.InputMaskingLayer import *

def get_svhn_model():
    model = Sequential()
    model.add(InputMaskingLayer(0.2))

    reNet_hidden_size = 5 #256
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2,
            is_first_layer=True))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.2))

    model.add(tf.keras.layers.Flatten())
    fully_conn_hidden_size = 20 #4096
    model.add(tf.keras.layers.Dense(fully_conn_hidden_size, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(fully_conn_hidden_size, activation='relu', name='features'))
    model.add(tf.keras.layers.Dropout(0.2))

    num_classes = 10
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

    return model
