import numpy as np
from keras.utils import to_categorical

from SimpleReNet import *


#model hyperparmeters:
w_p = 2
h_p = 2
reNet_hidden_size = 1
fully_conn_hidden_size = 1
num_classes = 2


#image parameters:
num_samples = 60
img_width = 10
img_height = 10
number_of_channels = 1


def simple_data_x():
    x = np.zeros((num_samples, img_width, img_height,
            number_of_channels), dtype=np.uint8)
    x[num_samples // 2:] = np.ones(
                (img_width, img_height, number_of_channels))
    return x


def simple_data_y():
    y = np.zeros((num_samples), dtype=np.uint8)
    y[num_samples // 2:] = 1
    return y


x = simple_data_x()
y = simple_data_y()
y = to_categorical(y, num_classes)


model = SimpleReNet([[w_p, h_p]],
        reNet_hidden_size, fully_conn_hidden_size, num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam',
        metrics=['categorical_accuracy'])
model.fit(x, y, epochs=100, shuffle=False, validation_data=(x, y))
