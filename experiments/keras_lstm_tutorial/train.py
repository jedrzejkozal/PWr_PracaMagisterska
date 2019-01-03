#source: http://adventuresinmachinelearning.com/keras-lstm-tutorial/

from keras.callbacks import ModelCheckpoint

from KerasBatchGenerator import KerasBatchGenerator
from model_configuration import *
from utils import *
from config import *


train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

model = configure_model(vocabulary, num_steps)

checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50

model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                    validation_data=valid_data_generator.generate(),
                    validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
model.save(data_path + "final_model.hdf5")
