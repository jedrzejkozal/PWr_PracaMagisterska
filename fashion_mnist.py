import os
import numpy as np

from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import l1, l2

from ReNet.Models.ReNetLayer import *
import sys
script_path = os.path.dirname(os.path.abspath(__file__))
modfied_path = os.path.join(script_path ,'ModifiedReNet/')
sys.path.append(modfied_path)
from Models.HilbertLayer.HilbertLayer import *
from Models.ModifiedReNetLayer import *
from ReNet.Utils.ReduceImbalance import *

num_classes = 10
img_rows, img_cols = 28, 28
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

def normalize(matrix):
    mu = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0)
    return (matrix - mu) / std

x_train_data = normalize(x_train)
x_test_data = normalize(x_test)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

print("y_train bincount: ", np.bincount(np.squeeze(y_train)))
print("y_test bincount: ", np.bincount(np.squeeze(y_test)))

# convert class vectors to binary class matrices
y_train_data = to_categorical(y_train, num_classes)
y_test_data = to_categorical(y_test, num_classes)

from PIL import Image

def resize_data(data, new_size):
    num_samples = data.shape[0]
    resized = np.zeros((num_samples,)+new_size+(1,), dtype=data.dtype)

    for i in range(num_samples):
        img = Image.fromarray(np.squeeze(data[i]))
        resized_img = np.asarray(img.resize(new_size))
        resized[i] = np.expand_dims(resized_img, axis=3)

    return resized

new_size = (32, 32)
x_train_data = resize_data(x_train_data, new_size)
x_test_data = resize_data(x_test_data, new_size)
print(x_train_data.shape)
print(x_test_data.shape)


def get_reNet(lr=0.001, dense_reg=l1(0.0000001), softmax_reg=l2(0.0000001)):
    model = Sequential()

    reNet_hidden_size = 256
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))

    model.add(Flatten())
    fully_conn_hidden_size = 4096
    model.add(Dense(fully_conn_hidden_size, activation='relu', activity_regularizer=dense_reg))
    model.add(Dropout(0.1))

    num_classes = 10
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=softmax_reg))

    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr),
            metrics=['categorical_accuracy'])

    return model

def get_modif_reNet(lr=0.001, dense_reg=None, softmax_reg=None):
    model = Sequential()

    reNet_hidden_size = 256
    model.add(HilbertLayer())
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ModifiedReNetLayer(4, reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))


    model.add(Flatten())
    fully_conn_hidden_size = 4096
    model.add(Dense(fully_conn_hidden_size, activation='relu', activity_regularizer=dense_reg))
    model.add(Dropout(0.1))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=softmax_reg))

    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr),
            metrics=['categorical_accuracy']
        )

    return model

def get_conv(lr=0.001, dense_reg=None, softmax_reg=None):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(96, kernel_size=(2, 2),
                 activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', activity_regularizer=dense_reg))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=softmax_reg))

    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr),
            metrics=['categorical_accuracy']
        )

    return model

results = {}

for learning_rate in [0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001]:
    print("learning_rate = ", learning_rate)
    model = get_modif_reNet(lr=learning_rate, dense_reg=None, softmax_reg=None)

    x_train_single_ex = x_train_data[0:1]
    y_train_single_ex = y_train_data[0:1]
    model.fit(x_train_single_ex, y_train_single_ex, epochs=1)
    #model.summary()

    datagen = ImageDataGenerator(width_shift_range=[-2.0, 0.0, 2.0])
    datagen.fit(x_train_data)

    batch_size = 32
    history = model.fit_generator(datagen.flow(x_train_data, y_train_data,
        batch_size=batch_size),
        epochs=20,
        steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
        validation_data=(x_test_data, y_test_data),
        callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
                   ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
            ]
    )
    loss, acc = tuple(model.evaluate(x_test, y_test, batch_size=batch_size))
    print("learning_rate = ", learning_rate)
    print("best test loss", loss)
    print("best test acc: ", acc)

    results[learning_rate] = tuple(loss, acc)
    print(results)

print(results)
