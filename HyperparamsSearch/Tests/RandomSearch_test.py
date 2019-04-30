from keras import Sequential
import pytest

from RandomSearch import *


class ModelMock(ModelIfc):

    def __init__(self):
        self.used_hyperparams = []


    def train(self, hyperparams):
        self.used_hyperparams.append(hyperparams)


    def evaluate(self, *hyperparams):
        return np.array([0.9, 0.8])


class TestRandomSearch(object):

    @pytest.fixture
    def sut_1(self):
        return RandomSearch(num_try=1)


    def test_type_of_model_is_incorrect(self, sut_1):
        model_arg = Sequential()
        hyperparams_arg = {"lr": [0.001, 0.0001],
                            "dropout": [0.1, 0.5]}
        with pytest.raises(TypeError) as err:
            result = sut_1.search_hyperparms(model_arg, hyperparams_arg)
            assert result is None

        assert "invalid model type" in str(err.value)


    def test_type_of_model_is_correct(self, sut_1):
        model_arg = ModelMock()
        hyperparams_arg = {"lr": [0.001, 0.0001],
                            "dropout": [0.1, 0.5]}
        try:
            sut_1.search_hyperparms(model_arg, hyperparams_arg)
        except:
            pytest.fail("Exception occured")


    def test_num_try_1_used_hyperparms_all_keys_have_len_1_lists(self, sut_1):
        model_arg = ModelMock()
        boundries_arg = {"lr": [0.001, 0.0001], #draw in log scale
                            "dropout": [0.1, 0.5]} #draw in normal scale
                            #how to do this? get functions instaed of borders
        assert True
