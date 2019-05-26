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
from sklearn.model_selection import StratifiedKFold

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


def get_reNet(lr=0.001, dense_reg=l1(0.0000001), softmax_reg=l2(0.0000001),
        reNet_hidden_size = 128, fully_conn_hidden_size = 4096):
    model = Sequential()

    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))
    model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
            use_dropout=True, dropout_rate=0.1))

    model.add(Flatten())
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

"""
results = {}

for hidden_size in [128, 256, 512, 1024, 2048, 4096]:
    learning_rate = 0.001
    reg = 0.0000001
    model = get_reNet(lr=learning_rate, dense_reg=l1(reg), softmax_reg=l2(reg), fully_conn_hidden_size=hidden_size)

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
                   ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)])

    loss, acc = tuple(model.evaluate(x_test_data, y_test_data, batch_size=batch_size))
    print("fully_conn_hidden_size = ", hidden_size)
    print("best test loss", loss)
    print("best test acc: ", acc)

    results[hidden_size] = tuple([loss, acc])
    print(results)

print(results)
"""

def train_on_fold(fold_num):
    x = np.vstack([x_train_data, x_test_data])
    y = np.hstack([convert_from_one_hot_to_labels(y_train_data), convert_from_one_hot_to_labels(y_test_data)])

    train_indexes, test_indexes = get_splits(x, y)
    indexes = train_indexes, test_indexes
    x_train, y_train, x_test, y_test = get_fold(fold_num, x, y, indexes)

    model = get_reNet()
    loss, acc = test_model_on_fold(model, x_train, y_train, x_test, y_test)
    print("fold_num = ", fold_num)
    print("test loss", loss)
    print("test acc: ", acc)


def get_fold(fold_num, x, y, indexes):
    fold_num = fold_num-1 #indexing from 0

    train_indexes, test_indexes = indexes

    x_train = x[train_indexes[fold_num]]
    y_train = y[train_indexes[fold_num]]
    x_test = x[test_indexes[fold_num]]
    y_test = y[test_indexes[fold_num]]

    x_train, y_train = undersample_to_lowest_cardinality_class(x_train, y_train)

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


def get_splits(x, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    train_indexes = []
    test_indexes = []
    for train_index, test_index in skf.split(x, y):
        print("train_index: ", train_index)
        print("test_index: ", test_index)
        train_indexes.append(train_index)
        test_indexes.append(test_index)

    return train_indexes, test_indexes


def test_model_on_fold(model, x_train, y_train, x_test, y_test):
    x_train_single_ex = x_train[0:1]
    y_train_single_ex = y_train[0:1]
    model.fit(x_train_single_ex, y_train_single_ex, epochs=1)

    #use multi-GPU model:
    #number_of_GPUs = 4
    #model = multi_gpu_model(model, gpus=number_of_GPUs)
    #model.compile(loss='categorical_crossentropy',
    #        optimizer=Adam(lr=0.001),
    #        metrics=['categorical_accuracy'])

    datagen = ImageDataGenerator(width_shift_range=[-2.0, 0.0, 2.0])
    datagen.fit(x_train)

    batch_size = 32
    model.fit_generator(datagen.flow(x_train, y_train,
            batch_size=batch_size),
            epochs=50,
            steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
            validation_data=(x_test, y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1, restore_best_weights=True),
                   ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1)
            ]
        )
    loss, acc = tuple(model.evaluate(x_test, y_test, batch_size=batch_size))
    return loss, acc


if __name__ == "__main__":
    train_on_fold(2)
