from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.regularizers import l1, l2

from ReNet.Models.ReNetLayer import *
from ModifiedReNet.Models.HilbertLayer.HilbertLayer import *
from ModifiedReNet.Models.ModifiedReNetLayer import *


def get_natural_img_reNet(lr=0.001, dense_reg=l1(0.00000001), softmax_reg=l2(0.00000001)):
    model = Sequential()
    num_classes = 8

    reNet_hidden_size = 128
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

def get_natural_img_modif_reNet(lr=0.001, dense_reg=l1(0.000000001), softmax_reg=l2(0.000000001), reNet_hidden_size=128, fully_conn_hidden_size=256):
    model = Sequential()
    num_classes = 8

    model.add(HilbertLayer())
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(1, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(1, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(1, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(1, reNet_hidden_size,
         use_dropout=True, dropout_rate=0.1))


    model.add(Flatten())
    model.add(Dense(fully_conn_hidden_size, activation='relu', activity_regularizer=dense_reg))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=softmax_reg))

    model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=lr),
        metrics=['categorical_accuracy'])

    return model
