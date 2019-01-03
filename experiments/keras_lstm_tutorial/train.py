#source: http://adventuresinmachinelearning.com/keras-lstm-tutorial/

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from KerasBatchGenerator import KerasBatchGenerator
from utils import *
from config import *


def configure_model(vocabulary, hidden_size, num_steps):
    model = Sequential()
    model.add(Embedding(vocabulary, hidden_size, input_length=num_steps))
    model.add(LSTM(hidden_size, return_sequences=True))
    model.add(LSTM(hidden_size, return_sequences=True))
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(vocabulary)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

    print(model.summary())
    return model



train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

hidden_size = 500
use_dropout=True

model = configure_model(vocabulary, hidden_size, num_steps)

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50

model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
model.save(data_path + "final_model.hdf5")
