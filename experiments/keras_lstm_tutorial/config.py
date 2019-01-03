import os
from utils import load_data


repo_path = os.path.dirname(os.path.realpath(__file__))
data_path = repo_path + "/../datasets/simple-examples/data"

train_data, valid_data, test_data, vocabulary, reversed_dictionary = load_data(data_path)

num_steps = 30
batch_size = 20
