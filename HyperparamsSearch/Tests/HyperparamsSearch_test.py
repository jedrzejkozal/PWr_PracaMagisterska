from keras import Sequential
import numpy as np
import pytest

from HyperparamsSearch import *


class ModelMock(ModelIfc):

    def __init__(self):
        self.used_hyperparams = {}

    def train(self, hyperparams):
        for hyperparam_name, value in hyperparams.items():
            self.__add_used(hyperparam_name, value)


    def __add_used(self, hyperparam_name, value):
        try:
            if self.used_hyperparams[hyperparam_name].count(value) == 0:
                self.used_hyperparams[hyperparam_name].append(value)
        except:
            self.used_hyperparams[hyperparam_name] = [value]


    def check_used_hyperparms(self, hyperparams):
        assert hyperparams == self.used_hyperparams


    def evaluate(self, *hyperparams):
        return np.array([0.9, 0.8])



class TestHyperparamsSearch(object):

    @pytest.fixture
    def sut(self):
        return HyperparamsSearch()


    def test_type_of_model_is_incorrect(self, sut):
        model_arg = Sequential()
        hyperparams_arg = {"lr": [0.001, 0.0001],
                            "dropout": [0.1, 0.5]}
        with pytest.raises(TypeError) as err:
            result = sut.search_hyperparms(model_arg, hyperparams_arg)
            assert result is None

        assert "invalid model type" in str(err.value)


    def test_type_of_model_is_correct(self, sut):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": [0.001, 0.0001],
                            "dropout": [0.1, 0.5]}
        try:
            sut.search_hyperparms(model_arg, hyperparams_arg)
        except:
            pytest.fail("Exception occured")


    def test_all_hyperparams_used(self, sut):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": [0.001, 0.0001],
                            "dropout": [0.1, 0.5]}
        sut.search_hyperparms(model_arg, hyperparams_arg)
        model_arg.check_used_hyperparms(hyperparams_arg)
