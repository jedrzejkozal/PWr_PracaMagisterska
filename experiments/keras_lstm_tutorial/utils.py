import os
import collections
import tensorflow as tf


def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()


def build_vocab(filename):
    data = read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def file_to_word_ids(filename, word_to_id):
    data = read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]


def get_data_paths(data_path):
    train_path = os.path.join(data_path, "ptb.train.txt")
    valid_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")
    return train_path, valid_path, test_path


def convert_text_data_to_list_of_integers(word_to_id):
    reversed_dictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    return reversed_dictionary


def read_data_as_ids(word_to_id, train_path, valid_path, test_path):
    train_data = file_to_word_ids(train_path, word_to_id)
    valid_data = file_to_word_ids(valid_path, word_to_id)
    test_data = file_to_word_ids(test_path, word_to_id)

    return train_data, valid_data, test_data


def build_complete_vocabulary(word_to_id, train_path, valid_path, test_path):
    train_data, valid_data, test_data = read_data_as_ids(word_to_id,
        train_path, valid_path, test_path)
    vocabulary = len(word_to_id)
    return train_data, valid_data, test_data, vocabulary


def load_data(data_path):
    train_path, valid_path, test_path = get_data_paths(data_path)
    # build the complete vocabulary, then convert text data to list of integers
    word_to_id = build_vocab(train_path)
    train_data, valid_data, test_data, vocabulary = build_complete_vocabulary(word_to_id,
        train_path, valid_path, test_path)
    reversed_dictionary = convert_text_data_to_list_of_integers(word_to_id)


    #print(train_data[:5])
    #print(word_to_id)
    #print(vocabulary)
    #print(" ".join([reversed_dictionary[x] for x in train_data[:10]]))
    return train_data, valid_data, test_data, vocabulary, reversed_dictionary
