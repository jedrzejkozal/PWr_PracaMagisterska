from config import *
from keras.models import load_model
from KerasBatchGenerator import KerasBatchGenerator
import numpy as np


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
