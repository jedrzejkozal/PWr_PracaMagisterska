import pytest
import numpy as np
from keras import Sequential

from RandomSearch import *


class ModelMock(ModelIfc):

    def __init__(self):
        self.used_hyperparams = {}


    def train(self, hyperparams):
        for hyperparam_name, value in hyperparams.items():
            self.__add_used(hyperparam_name, value)


    def __add_used(self, hyperparam_name, value):
        try:
            self.used_hyperparams[hyperparam_name].append(value)
        except:
            self.used_hyperparams[hyperparam_name] = [value]


    def evaluate(self, *hyperparams):
        return np.array([0.9, 0.8])


    def check_used_hyperparms(self, should_use):
        assert should_use == self.used_hyperparams


class HyperparamsGenerator(object):

    def __init__(self):
        self.called = -1
        self.hyperparams = [3.1415, 2.71, 42, 21.37]


    def __call__(self):
        self.called += 1
        return self.hyperparams[self.called]


class TestRandomSearch(object):


    @pytest.fixture
    def sut_1_try(self):
        return RandomSearch(num_try=1)


    @pytest.fixture
    def sut_2_try(self):
        return RandomSearch(num_try=2)


    def test_type_of_model_is_incorrect(self, sut_1_try):
        model_arg = Sequential()
        hyperparams_arg = {"lr": [0.001, 0.0001],
                            "dropout": [0.1, 0.5]}
        with pytest.raises(TypeError) as err:
            result = sut_1_try.search_hyperparms(model_arg, hyperparams_arg)
            assert result is None

        assert "invalid model type" in str(err.value)


    def test_type_of_model_is_correct(self, sut_1_try):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": lambda : 42,
                            "dropout": lambda : 42}
        try:
            sut_1_try.search_hyperparms(model_arg, hyperparams_arg)
        except:
            pytest.fail("Exception occured")


    def test_num_try_1_used_hyperparms_are_correct(self, sut_1_try):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": lambda : 42,
                        "dropout": lambda : 3.1415}
        sut_1_try.search_hyperparms(model_arg, hyperparams_arg)
        model_arg.check_used_hyperparms({"lr": [42],
                                        "dropout":[3.1415]})


    def test_num_try_2_used_hyperparms_are_correct(self, sut_2_try):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": lambda : 42,
                        "dropout": lambda : 3.1415}
        sut_2_try.search_hyperparms(model_arg, hyperparams_arg)
        model_arg.check_used_hyperparms({"lr": [42, 42],
                                        "dropout": [3.1415, 3.1415]})


    def test_num_try_2_params_from_same_generator_used(self, sut_2_try):
        model_arg = ModelMock()
        generator = HyperparamsGenerator()
        hyperparams_arg = {"lr": generator,
                        "dropout": generator}
        sut_2_try.search_hyperparms(model_arg, hyperparams_arg)
        model_arg.check_used_hyperparms({"lr": [3.1415, 42],
                                        "dropout": [2.71, 21.37]})


    def test_num_try_2_params_from_separate_generator_used(self, sut_2_try):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": HyperparamsGenerator(),
                        "dropout": HyperparamsGenerator()}
        sut_2_try.search_hyperparms(model_arg, hyperparams_arg)
        model_arg.check_used_hyperparms({"lr": [3.1415, 2.71],
                                        "dropout": [3.1415, 2.71]})
