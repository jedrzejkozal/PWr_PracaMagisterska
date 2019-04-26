import os
import numpy as np

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, TensorBoard, LambdaCallback
from keras.optimizers import Adam
from keras.models import Sequential

from Models.ReNetLayer import *
from Utils.ReduceImbalance import *

model = Sequential()

reNet_hidden_size = 20
model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
        use_dropout=True, dropout_rate=0.1,
        is_first_layer=True))
model.add(ReNetLayer([[2, 2]], reNet_hidden_size,
        use_dropout=True, dropout_rate=0.1))

model.add(Flatten())
fully_conn_hidden_size = 1024
model.add(Dense(fully_conn_hidden_size, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(fully_conn_hidden_size, activation='relu', name='features'))
model.add(Dropout(0.1))

num_classes = 2
model.add(Dense(num_classes, activation='softmax'))


model.compile(loss='categorical_crossentropy',
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=10.0**-8.0),
        metrics=['categorical_accuracy']
        )

num_samples = 5216
datagen = ImageDataGenerator()
iterator = datagen.flow_from_directory("/home/jkozal/Dokumenty/PWr/magisterka/magisterka/datasets/chest_xray/chest_xray/train", batch_size=num_samples)
batch_input, batch_labels = next(iterator)
print("before undersampling: ", batch_input.shape, batch_labels.shape)

normal_indexes = np.argwhere(batch_labels[:,1] == 0).flatten()
pneumonia_indexes = np.argwhere(batch_labels[:,1] == 1).flatten()

normal_chosen = np.random.choice(normal_indexes, 1000)
pneumonia_chosen = np.random.choice(pneumonia_indexes, 1000)

x_train = np.vstack([batch_input[normal_chosen], batch_input[pneumonia_chosen]])
y_train = np.vstack([batch_labels[normal_chosen], batch_labels[pneumonia_chosen]])
print("after undersampling: ", x_train.shape, y_train.shape)


model.fit(batch_input[0:1], batch_labels[0:1], epochs=1)
num_samples = 5216
batch_size = 32
model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=10,
        callbacks=[EarlyStopping(monitor='val_loss', patience=20, verbose=1)
            ]
    )
