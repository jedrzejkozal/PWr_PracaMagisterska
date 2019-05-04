from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

from ModelIfc import *


class KerasAdapter(ModelIfc):

    def __init__(self, dataset, model, epochs):
        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]
        self.model = model
        self.epochs = epochs


    def train(self, hyperparams):
        self.model.compile(loss='categorical_crossentropy',
                optimizer=Adam(lr=hyperparams["lr"]),
                metrics=['categorical_accuracy'])
        self.model.fit(x=self.x_train,
                y=self.y_train,
                batch_size=32,
                epochs=self.epochs,
                validation_data=(self.x_test, self.y_test),
                callbacks=[EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        verbose=1,
                                        restore_best_weights=True)])


    def evaluate(self, hyperparams):
        return self.model.evaluate(x=self.x_test, y=self.y_test)
