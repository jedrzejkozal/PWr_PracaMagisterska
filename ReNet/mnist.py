import os
import numpy as np

from keras.utils import to_categorical
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
from keras.optimizers import Adam
from os import makedirs
from os.path import exists, join
from shutil import rmtree

from Models.MnistReproduction.MnistModel import *


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

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)

print("y_train bincount: ", np.bincount(np.squeeze(y_train)))
print("y_test bincount: ", np.bincount(np.squeeze(y_test)))

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#just for testing
x_train = x_train[:10]
y_train = y_train[:10]
x_test = x_test[:10]
y_test = y_test[:10]


datagen = ImageDataGenerator(width_shift_range=[-2.0, 0.0, 2.0])
datagen.fit(x_train)


model = get_mnist_model()
model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10.0**-8.0),
        metrics=['categorical_accuracy'])


#just for model to figure out what is the shape of input tensors
#workaround for how keras fit_generator works
x_train_single_ex = x_train[0:1]
y_train_single_ex = y_train[0:1]
model.fit(x_train_single_ex, y_train_single_ex, epochs=1)
model.summary()

batch_size = 30

def mask_input(x):
    print("mask_input call")
    mask = generate_mask_with_prob(0.2)

    tmp = np.copy(x)
    for i in range(x.shape[0]):
        tmp[i][mask] = -100.0
    return tmp

def generate_mask_with_prob(p):
    rand_matrix = get_random_binomial(p, get_patched_img_size())
    expand_matrix = get_expand_matrix(img_rows // 2)
    single_dim_mask = np.transpose(expand_matrix) @ rand_matrix @ expand_matrix

    all_dims = [single_dim_mask]
    mask = np.stack(all_dims, axis=2)

    return mask > 0.5


def get_patched_img_size():
    img_size = img_rows * img_cols
    return img_size // 4  # patch_size_x*patch_size_y = 2*2 = 4


def get_random_binomial(p, patched_img_size):
    rand_matrix = np.random.binomial(1, p, size=patched_img_size)
    return rand_matrix.reshape(img_rows // 2, img_cols // 2)


def get_expand_matrix(n):
    expand_matrix = np.zeros((n, 2*n)) #won't work for rectangular imgs
    for i in range(n):
        expand_matrix[i, 2*i:2*i+2] = 1
    return expand_matrix


num_epochs = 10
for i in range(num_epochs):
    print("epoch {}/{}".format(i+1, num_epochs))
    masked_x_train = mask_input(x_train)
    model.fit_generator(datagen.flow(masked_x_train, y_train, batch_size=batch_size),
            epochs=1,
            steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
            validation_data=(x_test, y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1)
                ]
        )
