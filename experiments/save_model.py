import os
import numpy as np

from keras import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten

import sys
sys.path.append('/home/jkozal/Dokumenty/PWr/magisterka/ReNet/Models')
from ReNetLayer import *


n = 20
x_train = np.zeros((n, 6, 6, 1))
y_train = np.zeros((n))

def get_model():
    model = Sequential()
    model.add(ReNetLayer([[2, 2]], 1,
            use_dropout=True, dropout_rate=0.1))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='relu'))
    return model

model = get_model()
model.compile(loss='mse',
        optimizer=Adam(lr=0.01),
        metrics=['categorical_accuracy'])


model.fit(x_train, y_train, epochs=1)
model.summary()

def save_weights(model, savedir):
    create_dir(savedir)
    for index, layer in enumerate(model.layers):
        weights = layer.get_weights()
        filedir = os.path.join(savedir, str(index+1))
        np.save(filedir, weights)


def create_dir(dirname):
    try:
        os.mkdir(dirname)
    except:
        return


dir = "model_weights"
save_weights(model, dir)
del model

def load_weights(model, loaddir):
    filelist = os.listdir(loaddir)
    filelist.sort()
    for weights_file, layer in zip(filelist, model.layers):
        filedir = os.path.join(loaddir, weights_file)
        weights = np.load(filedir)
        layer.set_weights(weights)


model = get_model()
model.compile(loss='mse',
        optimizer=Adam(lr=0.01),
        metrics=['categorical_accuracy'])
x_train_single_ex = x_train[0:1]
y_train_single_ex = y_train[0:1]
model.fit(x_train_single_ex, y_train_single_ex, epochs=1)

load_weights(model, dir)
model.evaluate(x_train, y_train)
