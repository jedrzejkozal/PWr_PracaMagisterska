import os
import numpy as np

from keras.utils import to_categorical
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
from keras.optimizers import Adam
from sklearn.decomposition import PCA
from shutil import rmtree
from os import makedirs

from ReNet.Models.Cifar10Reproduction.Cifar10Model import *
from Utils.InputNormalization import *
from Utils.Masking import *


#image parameters:
num_classes = 10
img_rows, img_cols = 32, 32
num_channels = 3
(x_train, y_train), (x_test, y_test) = cifar10.load_data()


x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255


# ZCA
x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols*3)
x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols*3)

pca = PCA(n_components=400, random_state=0, svd_solver='randomized')
pca.fit(x_train)

W = np.dot(pca.components_.T @ np.diag(1. / pca.singular_values_), pca.components_) * np.sqrt(x_train.shape[0]) * 64

def ZCA(x):
    return np.dot(x, W) + 128

x_train = x_train - pca.mean_
x_test = x_test - pca.mean_
for i in range(x_train.shape[0]):
    x_train[i] = ZCA(x_train[i])
for i in range(x_test.shape[0]):
    x_test[i] = ZCA(x_test[i])

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3)

import matplotlib.pyplot as plt
plt.figure()
for xx in range(25):
    plt.subplot(5,5,xx+1)
    plt.imshow(x_train[xx].astype('int32'))
#plt.show()

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


datagen = ImageDataGenerator(width_shift_range=[-2.0, 0.0, 2.0],
                horizontal_flip=True,
                vertical_flip=False)
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

batch_size = 32
num_epochs = 50
masking = Masking(img_rows, img_cols, num_channels)
for i in range(num_epochs):
    print("epoch {}/{}".format(i+1, num_epochs))
    masked_x_train = masking.mask_input(x_train)
    model.fit_generator(datagen.flow(masked_x_train, y_train,
            batch_size=batch_size),
            epochs=1,
            steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
            validation_data=(x_test, y_test),
            callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1)
                ]
        )
