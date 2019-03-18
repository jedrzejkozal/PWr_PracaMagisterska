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

from Utils.SaveResults import *
from Utils.TensorBoardSaveSplits import *
from Utils.SaveTensorBoardSprite import *
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


# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("x_train: ", x_train.shape)
print("y_train: ", y_train.shape)

x_train_single_ex = x_train[0:1]
y_train_single_ex = y_train[0:1]


x_train = x_train[:50000]
y_train = y_train[:50000]


#just for testing
#x_train = x_train[:100]
#y_train = y_train[:100]
#x_test = x_test[:100]
#y_test = y_test[:100]

log_dir = 'TensorBoard_logs'
rmtree(log_dir, ignore_errors=True)

makedirs(log_dir)
save_sprites(x_test, log_dir)


# save class labels to disk to color data points in TensorBoard accordingly
with open(join(log_dir, 'metadata.tsv'), 'w') as f:
    np.savetxt(f, y_test)


shift = 3
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
datagen.fit(x_train)


model = get_mnist_model()
model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10.0**-8.0),
        metrics=['categorical_accuracy'])


#just for model to figure out what is the shape of input tensors
#workaround for how keras fit_generator works
model.fit(x_train_single_ex, y_train_single_ex, epochs=1)
model.summary()

batch_size = 30
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=10,
        steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
        validation_data=(x_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                LambdaCallback(on_epoch_end=lambda x, y: model.layers[0].generate_mask()),
                TensorBoardSaveSplits(log_dir=log_dir,
                        splits_size=[28,28],
                        splits_path='sprite.png',
                        batch_size=batch_size,
                        histogram_freq=1,
                        write_images=True,
                        write_grads=False,
                        embeddings_freq=1,
                        embeddings_layer_names=['features'],
                        embeddings_metadata='metadata.tsv',
                        embeddings_data=x_test)
            ]
    )

score = model.evaluate(x_test, y_test, verbose=0)
