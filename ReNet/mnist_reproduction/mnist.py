import os
import numpy as np
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import sys
sys.path.append('../Utils')
from SaveResults import *

from MnistReproduction import *


#model hyperparmeters:
w_p = 2
h_p = 2
reNet_hidden_size = 1
fully_conn_hidden_size = 1
num_classes = 2


#image parameters:
num_classes = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("x_train: ", x_train.shape)
print("x_train_single: ", x_train[0:1].shape)
print("y_train: ", y_train.shape)
print("y_train_single:", y_train[0:1].shape)

x_train_single_ex = x_train[0:1]
y_train_single_ex = y_train[0:1]

shift = 3
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
datagen.fit(x_train)


model = MnistReproduction()
model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['categorical_accuracy'])

#just for model to figure out what is the shape of input tensors
#workaround for how keras fit_generator works
model.fit(x_train_single_ex, y_train_single_ex,
                epochs=1,
                validation_data=(x_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
            )

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                epochs=20,
                steps_per_epoch=60,
                validation_data=(x_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1)]
            )


path = os.path.dirname(os.path.realpath(__file__))
print("path: ", path)
save = SaveResults(history, path)
