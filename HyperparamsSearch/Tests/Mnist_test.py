from keras.layers import Dense, Dropout, Flatten
from keras.datasets import mnist
from keras import Sequential
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping



from HyperparamsSearch import *


class MnistModel(ModelIfc):

    def __init__(self):
        self.num_classes = 10
        self.img_rows, self.img_cols = 28, 28
        self.__load_mnist()


    def __load_mnist(self):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0],
                    self.img_rows, self.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0],
                    self.img_rows, self.img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        y_train = to_categorical(y_train, self.num_classes)
        y_test = to_categorical(y_test, self.num_classes)

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test


    def __prepare_model(self, hyperparams):
        self.model = Sequential()
        self.model.add(Flatten())
        self.model.add(Dense(hyperparams["neurons"], activation='relu'))
        self.model.add(Dropout(hyperparams["dropout"]))
        self.model.add(Dense(hyperparams["neurons"], activation='relu'))
        self.model.add(Dropout(hyperparams["dropout"]))
        self.model.add(Dense(self.num_classes, activation='softmax'))


    def train(self, hyperparams):
        self.__prepare_model(hyperparams)
        self.model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=hyperparams["lr"]),
                metrics=['categorical_accuracy'])
        self.model.fit(x=self.x_train,
                y=self.y_train,
                batch_size=30,
                epochs=1,
                validation_data=(self.x_test, self.y_test),
                callbacks=[EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        verbose=1)])


    def evaluate(self, hyperparams):
        return self.model.evaluate(x=self.x_test, y=self.y_test)


h = HyperparamsSearch()
hyperparams_arg = {"lr": [0.001],
                    "neurons": [2],
                    "dropout": [0.1, 0.5]}
modelAdapter = MnistModel()
h.search_hyperparms(modelAdapter, hyperparams_arg)
