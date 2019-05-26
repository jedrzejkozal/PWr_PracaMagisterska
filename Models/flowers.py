from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Floatten, Dense, Dropout
from keras.regularizers import l1, l2

from ReNet.Models.ReNetLayer import *
from ModifiedReNet.Models.HilbertLayer import *
from ModifiedReNet.Models.ModifiedReNetLayer import *


def get_flowers_reNet(lr=0.001, dense_reg=l1(0.00000001), softmax_reg=l2(0.00000001)):
    model = Sequential()

    reNet_hidden_size = 256
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
        use_dropout=True, dropout_rate=0.1))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
        use_dropout=True, dropout_rate=0.1))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
        use_dropout=True, dropout_rate=0.1))

    model.add(Flatten())
    fully_conn_hidden_size = 4096
    model.add(Dense(fully_conn_hidden_size, activation='relu', activity_regularizer=dense_reg))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=softmax_reg))

    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['categorical_accuracy'])

    return model

def get_flowers_modif_reNet(lr=0.001, dense_reg=l1(0.00000001), softmax_reg=l2(0.0000001)):
    model = Sequential()

    reNet_hidden_size = 256
    model.add(HilbertLayer())
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.2))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.2))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.2))

    model.add(Flatten())
    fully_conn_hidden_size = 1024
    model.add(Dense(fully_conn_hidden_size, activation='relu', activity_regularizer=dense_reg))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=softmax_reg))

    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['categorical_accuracy'])

    return model
