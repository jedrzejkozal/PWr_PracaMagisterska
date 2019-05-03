from keras.models import Sequential, Model
from keras.optimizers import Adam

from Tests.IdentityReNetLayer import *

import pytest
import numpy as np


x = np.array([[[1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]]]).reshape(1, 4, 4, 1)
print("x: ", x)
print("x.shape: ", x.shape)
y = np.array([[[1, 2, 5, 6, 3, 4, 7, 8],
                [3, 4, 7, 8, 1, 2, 5, 6]],
                [[9,10,13,14,11,12,15,16],
                [11,12,15,16,9,10,13,14]]]).reshape(1, 2, 2, 8)
print("y: ", y)
print("y.shape: ", y.shape)


model = Sequential()
model.add(IdentityReNetLayer([[2, 2]], 4))


model.compile(loss='mse',
        optimizer=Adam(),
        metrics=['categorical_accuracy'])
model.fit(x, y,
        epochs=1,
        batch_size=1)

print("layer:")
print(model.layers[0].LSTM_up_down)
print("weights: ")
print(model.layers[0].LSTM_up_down.get_weights())

def print_shapes(weights):
    print("shapes of weights:")
    for i in range(len(weights)):
        print(weights[i].shape)

def set_LSTM_identity_weights(lstm):
    weights = lstm.get_weights()
    print_shapes(weights)
    n = weights[1].shape[0]
    #forget gate - or at least i hope it's forget gate
    weights[0][:, :n] = np.zeros((n, n))

    #update gate
    weights[0][:, n:2*n] = np.identity(n)

    #output gate
    weights[0][:, 2*n:3*n] = np.identity(n)

    #tanh
    weights[0][:, 3*n:] = np.identity(n)

    weights[1] = np.zeros(weights[1].shape)

    lstm.set_weights(weights)


def set_LSTM_identity_weights_longer_matrix_select_first(lstm):
    weights = lstm.get_weights()
    print_shapes(weights)
    n = weights[1].shape[0]
    #forget gate - or at least I hope it's forget gate
    weights[0][:, :n] = np.zeros((2*n, n))

    #update gate
    weights[0][:n, n:2*n] = np.identity(n)
    weights[0][n:, n:2*n] = np.zeros((n, n))

    #output gate
    weights[0][:n, 2*n:3*n] = np.identity(n)
    weights[0][n:, 2*n:3*n] = np.zeros((n, n))

    #tanh
    weights[0][:n, 3*n:] = np.identity(n)
    weights[0][n:, 3*n:] = np.zeros((n, n))

    weights[1] = np.zeros(weights[1].shape)

    lstm.set_weights(weights)


def set_LSTM_identity_weights_longer_matrix_select_second(lstm):
    weights = lstm.get_weights()
    print_shapes(weights)
    n = weights[1].shape[0]
    #forget gate - or at least I hope it's forget gate
    weights[0][:, :n] = np.zeros((2*n, n))

    #update gate
    weights[0][:n, n:2*n] = np.zeros((n, n))
    weights[0][n:, n:2*n] = np.identity(n)

    #output gate
    weights[0][:n, 2*n:3*n] = np.zeros((n, n))
    weights[0][n:, 2*n:3*n] = np.identity(n)

    #tanh
    weights[0][:n, 3*n:] = np.zeros((n, n))
    weights[0][n:, 3*n:] = np.identity(n)

    weights[1] = np.zeros(weights[1].shape)

    lstm.set_weights(weights)


set_LSTM_identity_weights(model.layers[0].LSTM_up_down)
set_LSTM_identity_weights(model.layers[0].LSTM_down_up)
set_LSTM_identity_weights_longer_matrix_select_first(model.layers[0].LSTM_left_right)
set_LSTM_identity_weights_longer_matrix_select_first(model.layers[0].LSTM_right_left)



print("====================================")
x_test = np.array([[[0,1,2,3],[4,5,6,7]]])
inputs = Input(shape=x_test.shape[1:])
out = model.layers[0].LSTM_up_down(inputs)
model_test = Model(inputs=inputs, outputs=out)
model_test.compile(loss='mse',
        optimizer=Adam(),
        metrics=['categorical_accuracy'])
result = model_test.predict(x_test)
print("target: ", x_test)
print("prediction: ", result)
print("====================================")

print("====================================")
x_test = np.array([[[0,1,2,3],[4,5,6,7]]])
inputs = Input(shape=x_test.shape[1:])
out = model.layers[0].LSTM_down_up(inputs)
model_test = Model(inputs=inputs, outputs=out)
model_test.compile(loss='mse',
        optimizer=Adam(),
        metrics=['categorical_accuracy'])
result = model_test.predict(x_test)
print("target: ", x_test)
print("prediction: ", result)
print("====================================")

print("====================================")
x_test = np.array([[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]])
inputs = Input(shape=x_test.shape[1:])
out = model.layers[0].LSTM_left_right(inputs)
model_test = Model(inputs=inputs, outputs=out)
model_test.compile(loss='mse',
        optimizer=Adam(),
        metrics=['categorical_accuracy'])
result = model_test.predict(x_test)
print("target: ", x_test)
print("prediction: ", result)
print("====================================")

print("====================================")
x_test = np.array([[[0,1,2,3,4,5,6,7],[8,9,10,11,12,13,14,15]]])
inputs = Input(shape=x_test.shape[1:])
out = model.layers[0].LSTM_right_left(inputs)
model_test = Model(inputs=inputs, outputs=out)
model_test.compile(loss='mse',
        optimizer=Adam(),
        metrics=['categorical_accuracy'])
result = model_test.predict(x_test)
print("target: ", x_test)
print("prediction: ", result)
print("====================================")

result = model.predict(x)
print("target: ", y)
print("prediction: ", result)

assert np.isclose(y, result).all()
