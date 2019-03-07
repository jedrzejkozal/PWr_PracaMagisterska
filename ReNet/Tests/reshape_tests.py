from keras.layers import LSTM, Dense, Flatten
from keras.layers import Input, Reshape, Permute, concatenate
from keras.layers import Layer, Dropout, Masking
from keras.models import Sequential
import tensorflow as tf
import numpy as np
import keras

sess = tf.InteractiveSession()

#get_vertical_patches()
x = tf.constant([[ [[0],[1]],
                [[2],[3]],
                [[4],[5]],
                [[6],[7]],
                [[8],[9]],
                [[10],[11]],
                [[12],[13]],
                [[14],[15]],
                [[16],[17]],
                [[18],[19]]
                ]], dtype=tf.float32)
print("x: ", x)

r = Reshape((5,4,1))
reshaped = r(x)
print("reshaped: ", reshaped)
reshaped = tf.Print(reshaped, [reshaped], summarize=20, message="This is reshaped: ")
reshaped.eval()
print("reshaped.shape: ", reshaped.shape)

reshaped = tf.squeeze(reshaped, axis=3)

hidden_size=10
model = Sequential()
l = LSTM(hidden_size, return_sequences=True)
model.add(l)
model.add(Dense(1))
model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
re = np.array([[[0],[1],[2],[3]],[[4],[5],[6],[7]],[[8],[9],[10],[11]],[[12],[13],[14],[15]],[[16],[17],[18],[19]]], dtype=np.float32)
re_y = np.array([[[1], [2], [3], [4]],[[1], [2], [3], [4]],[[1], [2], [3], [4]],[[1], [2], [3], [4]],[[1], [2], [3], [4]]], dtype=np.float32)
model.fit(re, re_y,
          batch_size=1,
          epochs=1,
          verbose=1)
LSTM_output = l(reshaped)
LSTM_output = tf.Print(LSTM_output, [LSTM_output], summarize=20, message="This is l: ")
LSTM_output.eval()
