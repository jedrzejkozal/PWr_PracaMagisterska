import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K


batch_size = 128
num_classes = 10
epochs = 12

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('\nx_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples\n')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class SimpleConv(keras.Model):

    def __init__(self, num_classes=10):
        super(SimpleConv, self).__init__(name='conv')

        self.num_classes = num_classes

        self.conv1 = Conv2D(32, kernel_size=(3, 3),
                                activation='relu',
                                input_shape=input_shape)
        self.conv2 = Conv2D(64, (3, 3), activation='relu')
        self.pooling = MaxPooling2D(pool_size=(2, 2))
        self.dropout = Dropout(0.25)
        self.flatten = Flatten()

        self.dense1 = keras.layers.Dense(128, activation='relu')
        self.dense2 = keras.layers.Dense(num_classes, activation='softmax')


    def call(self, inputs):
        print("\n\nSimpleConv call")
        print("inputs: ", inputs, "\n\n")


        conv = self.conv1(inputs)
        conv = self.conv2(conv)
        conv = self.pooling(conv)
        conv = self.dropout(conv)
        conv = self.flatten(conv)

        x = self.dense1(conv)
        return self.dense2(x)


model = SimpleConv(num_classes=num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
