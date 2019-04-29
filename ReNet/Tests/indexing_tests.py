from keras.datasets import mnist

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cbook as cbook


num_classes = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
input_shape = (img_rows, img_cols)

fig, ax = plt.subplots()
im = ax.imshow(x_train[0])
ax.axis('off')


fig, ax = plt.subplots()
x_train[0, 5:7, :] = 255 #wybiera wiersz
im = ax.imshow(x_train[0])
ax.axis('off')

fig, ax = plt.subplots()
x_train[0, :, 5:7] = 255 #wybiera kolumnę
im = ax.imshow(x_train[0])
ax.axis('off')

#plt.show()


import tensorflow as tf
import numpy as np
import keras

sess = tf.InteractiveSession()

#get_vertical_patches()
x = tf.constant([ [[0],[1]],
                [[2],[3]],
                [[4],[5]],
                [[6],[7]],
                [[8],[9]],
                [[10],[11]],
                [[12],[13]],
                [[14],[15]],
                [[16],[17]],
                [[18],[19]]
                ], dtype=tf.float32)
print("x: ", x)
print("x.shape: ", x.shape)

#wybiera kolumnę
col = x[:, 1:]
print("col :", col)
print("col.shape :", col.shape)

tf.Print(col, [col], summarize=20, message="This is col: ")
col.eval()

#wybiera wiersz
row = x[3:5, :]
print("row :", row)
print("row.shape :", row.shape)
