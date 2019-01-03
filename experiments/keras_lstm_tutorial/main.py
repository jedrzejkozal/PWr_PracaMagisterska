#source: http://adventuresinmachinelearning.com/keras-lstm-tutorial/

import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Embedding, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np

from KerasBatchGenerator import KerasBatchGenerator
from utils import *

repo_path = os.path.dirname(os.path.realpath(__file__))
data_path = repo_path + "/../datasets/simple-examples/data"


args = parse_args(data_path)
if args.data_path:
    data_path = args.data_path


train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data(data_path)

num_steps = 30
batch_size = 20
train_data_generator = KerasBatchGenerator(train_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)
valid_data_generator = KerasBatchGenerator(valid_data, num_steps, batch_size, vocabulary,
                                           skip_step=num_steps)

hidden_size = 500
use_dropout=True


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
checkpointer = ModelCheckpoint(filepath=data_path + '/model-{epoch:02d}.hdf5', verbose=1)
num_epochs = 50
if args.run_opt == 1:
    model.fit_generator(train_data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs,
                        validation_data=valid_data_generator.generate(),
                        validation_steps=len(valid_data)//(batch_size*num_steps), callbacks=[checkpointer])
    model.save(data_path + "final_model.hdf5")
elif args.run_opt == 2:
    model = load_model(data_path + "/model-08.hdf5")
    dummy_iters = 40
    example_training_generator = KerasBatchGenerator(train_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
    print("Training data:")
    for i in range(dummy_iters):
        dummy = next(example_training_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_training_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps-1, :])
        true_print_out += reversed_dictionary[train_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
    # test data set
    dummy_iters = 40
    example_test_generator = KerasBatchGenerator(test_data, num_steps, 1, vocabulary,
                                                     skip_step=1)
    print("Test data:")
    for i in range(dummy_iters):
        dummy = next(example_test_generator.generate())
    num_predict = 10
    true_print_out = "Actual words: "
    pred_print_out = "Predicted words: "
    for i in range(num_predict):
        data = next(example_test_generator.generate())
        prediction = model.predict(data[0])
        predict_word = np.argmax(prediction[:, num_steps - 1, :])
        true_print_out += reversed_dictionary[test_data[num_steps + dummy_iters + i]] + " "
        pred_print_out += reversed_dictionary[predict_word] + " "
    print(true_print_out)
    print(pred_print_out)
