import os
import numpy as np

#from keras.utils import to_categorical
from keras.datasets import cifar10
#from keras.preprocessing.image import ImageDataGenerator
#from keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
#from keras.optimizers import Adam
#from shutil import rmtree
#from os import makedirs

#from Utils.SaveResults import *
#from Utils.TensorBoardSaveSplits import *
from Utils.InputNormalization import *
#from Models.Cifar10Reproduction.Cifar10Model import *

#from Utils.ImageGeneratorWithMasking import *


#image parameters:
num_classes = 10
img_rows, img_cols = 32, 32
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
image_index = 3
image = x_test[image_index]
im = ax.imshow(image)

import sys
np.set_printoptions(threshold=sys.maxsize)

# ZCA
x_test_red = x_test[:, :, :, 0]
x_test_green = x_test[:, :, :, 1]
x_test_blue = x_test[:, :, :, 2]

x_test_red = x_test_red.reshape(x_test_red.shape[0], img_rows*img_cols)
x_test_green = x_test_green.reshape(x_test_green.shape[0], img_rows*img_cols)
x_test_blue = x_test_blue.reshape(x_test_blue.shape[0], img_rows*img_cols)
del x_test
del x_train
del y_train

#x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols*3)


def calc_W(X):
    sigma = np.cov(np.transpose(X) @ X)
    evalues, evectors = np.linalg.eigh(sigma)
    del sigma

    number_of_evectors = evectors.shape[1]
    len_of_eigenvector = X.shape[0]
    eigenvectors = np.zeros((len_of_eigenvector, number_of_evectors))
    for i in range(number_of_evectors):
        eigenvectors[:, i] = X @ evectors[:,i]
    del evectors

    n_samples = X.shape[0]
    W = np.sqrt(n_samples-1) * eigenvectors @ np.diag(evalues**(-1/2)) @ eigenvectors.T
    print(W)
    #print(evalues**(-1/2))
    #print(evectors.T)
    #print(np.sqrt(n_samples-1))
    #print(evectors @ np.diag(evalues**(-1/2)) @ evectors.T)
    return W

# please note:
# in https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf matrix X is defined as (d, n)
# so with this notation result ZCA matrix with shape (n, d) is calculated as Y^T = X^T W^T
# so using notation with X as (n, d) Y = X W^T with W calculated for X defined as (d, n)
#def ZCA(X):
#    return X @ np.transpose(W)

W_red = calc_W(x_test_red)
#print("W_red: ", W_red)
W_green = calc_W(x_test_green)
W_blue = calc_W(x_test_blue)

#x_train = ZCA(x_train)
x_test_red = x_test_red @ np.transpose(W_red)
x_test_green = x_test_green @ np.transpose(W_green)
#x_test_blue = x_test_blue @ np.transpose(W_blue)

#x_test = ZCA(x_test)

x_test_red = x_test_red.reshape(x_test_red.shape[0], img_rows, img_cols, 1)
x_test_green = x_test_green.reshape(x_test_green.shape[0], img_rows, img_cols, 1)
x_test_blue = x_test_blue.reshape(x_test_blue.shape[0], img_rows, img_cols, 1)

x_test = np.concatenate([x_test_red, x_test_green, x_test_blue], axis=3)

#x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)

fig, ax = plt.subplots()
image = x_test[image_index]
im = ax.imshow(image)

fig, ax = plt.subplots()
image = np.concatenate([x_test_red[image_index], np.zeros(x_test_red[image_index].shape), np.zeros(x_test_red[image_index].shape)], axis=2)
im = ax.imshow(image)
plt.show()

"""
x_train = normalize(x_train)
x_test = normalize(x_test)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)
print("x_test: ", x_test.shape)
print("y_test: ", y_test.shape)


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#just for testing
#x_train = x_train[:100]
#y_train = y_train[:100]
#x_test = x_test[:100]
#y_test = y_test[:100]

log_dir = 'TensorBoard_cifar10_logs'
rmtree(log_dir, ignore_errors=True)
makedirs(log_dir)

shift = 3
datagen = ImageDataGenerator(width_shift_range=[-2.0, 0.0, 2.0],
                zca_whitening=True,
                horizontal_flip=True,
                vertical_flip=True)
#datagen = ImageDataGeneratorWithMasking(width_shift_range=shift,
#                height_shift_range=shift,
#                zca_whitening=True,
#                horizontal_flip=True,
#                vertical_flip=True)
datagen.fit(x_train)


model = get_cifar10_model()
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
history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                epochs=10,
                steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
                validation_data=(x_test, y_test),
                callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1),
                        #LambdaCallback(on_epoch_end=lambda x, y: model.layers[0].generate_mask()),
                        TensorBoardSaveSplits(log_dir=log_dir,
                                #splits_size=[28,28],
                                #splits_path='sprite.png',
                                batch_size=batch_size,
                                histogram_freq=1,
                                write_images=True,
                                write_grads=False,
                                #embeddings_freq=1,
                                #embeddings_layer_names=['features'],
                                #embeddings_metadata='metadata.tsv',
                                #embeddings_data=x_test
                                )
                ]
            )
"""
