from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist
from keras import Sequential
from keras.utils import to_categorical

from KerasAdapter import *


def get_mnist_model():
    num_classes = 10
    img_rows, img_cols = 28, 28

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0],
            img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0],
            img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    dataset = {"x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
            "y_test": y_test}

    model = Sequential()
    model.add(Flatten())
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(2, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    return KerasAdapter(dataset, model, 1)
