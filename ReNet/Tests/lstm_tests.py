from keras.models import Sequential
from keras.layers import LSTM
from keras.optimizers import Adam
from keras import backend as K

import numpy as np


#so this is basicaly how you can make LSTM that works as identity

num_features = 4

def constant_activation(x):
    return K.constant(1, shape=[num_features])



x = np.array([[[12, 11, 13, 14],
                [26, 27, 28, 15],
                [3, 3, 3, 6],
                [4, 4, 4, 8],
                [5, 5, 5, 9]]])
print("x: ", x)

model = Sequential()
model.add(LSTM(x.shape[2],
        activation=constant_activation,
        recurrent_activation='linear',
        return_sequences=True))

model.compile(loss='mse',
        optimizer=Adam(),
        metrics=['categorical_accuracy'])
model.fit(x, x,
        epochs=1000,
        batch_size=1)

weights = model.get_weights()
print("weights: ")
print(weights)

print("shapes")
for i in range(len(weights)):
    print(weights[i].shape)

#forget gate - or at least i hope it's forget gate
weights[0][:, :num_features] = np.zeros((num_features, num_features))
weights[1][:, :num_features] = np.zeros((num_features, num_features))

#update gate
weights[0][:, num_features:2*num_features] = np.identity(num_features)
weights[1][:, num_features:2*num_features] = np.zeros(num_features)

#output gate
weights[0][:, 2*num_features:3*num_features] = np.identity(num_features)
weights[1][:, 2*num_features:3*num_features] = np.zeros(num_features)

#tanh
weights[0][:, 3*num_features:] = np.identity(num_features)
weights[1][:, 3*num_features:] = np.zeros(num_features)

#bias
weights[2] = np.zeros(num_features*4)

print("after modifications: ")
print(weights)
model.set_weights(weights)

result = model.predict(x)
print("prediction: ", result)
print("target: ", x)
