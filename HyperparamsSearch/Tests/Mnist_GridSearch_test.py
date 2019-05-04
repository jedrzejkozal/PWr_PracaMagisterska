from GridSearch import *
from Tests.MnistModel import *


sut = GridSearch()
hyperparams_arg = {"lr": [0.01, 0.001]}
modelAdapter = get_mnist_model()
sut.search_hyperparms(modelAdapter, hyperparams_arg)
