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
print("x_train_single: ", x_train[0:1].shape)
print("y_train: ", y_train.shape)
print("y_train_single:", y_train[0:1].shape)

x_train_single_ex = x_train[0:1]
y_train_single_ex = y_train[0:1]


x_train = x_train[:50000]
y_train = y_train[:50000]

#just for testing
x_train = x_train[:100]
y_train = y_train[:100]
x_test = x_test[:100]
y_test = y_test[:100]

#image = tf.reshape(x_test, [-1, 28, 28, 1])
#tf.summary.image("image", image)

log_dir = './TensorBoard_logs'
rmtree(log_dir, ignore_errors=True)
makedirs(log_dir)



def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))


    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage

def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))

def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


to_visualise = x_test
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

import matplotlib.pyplot as plt
path_for_mnist_sprites =  os.path.join('TensorBoard_logs', 'mnistdigits.png')
print("path_for_mnist_sprites: ", path_for_mnist_sprites)
plt.imsave(path_for_mnist_sprites, sprite_image,cmap='gray')
plt.imshow(sprite_image, cmap='gray')






# save class labels to disk to color data points in TensorBoard accordingly
with open(join(log_dir, 'metadata.tsv'), 'w') as f:
    np.savetxt(f, y_test)


shift = 3
datagen = ImageDataGenerator(width_shift_range=shift, height_shift_range=shift)
datagen.fit(x_train)


model = get_model()
model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10.0**-8.0),
        metrics=['categorical_accuracy'])


#just for model to figure out what is the shape of input tensors
#workaround for how keras fit_generator works
model.fit(x_train_single_ex, y_train_single_ex, epochs=1)
model.summary()

batch_size = 30
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=2,#1000,
        steps_per_epoch=np.ceil(x_train.shape[0] / batch_size),
        validation_data=(x_test, y_test),
        callbacks=[EarlyStopping(monitor='val_loss', patience=5, verbose=1),
                #LambdaCallback(on_epoch_end=lambda x, y: model.input_masking.generate_mask()),
                TensorBoardSaveSplits(log_dir=log_dir,
                        splits_size=[28,28],
                        splits_path='mnistdigits.png',#path_for_mnist_sprites,
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
