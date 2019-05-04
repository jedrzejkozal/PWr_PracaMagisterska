import numpy as np

from RandomSearch import *
from Tests.MnistModel import *


sut = RandomSearch(num_try=2)
hyperparams_arg = {"lr": lambda : np.float_power([10.0], -10.0*np.random.random())}
modelAdapter = get_mnist_model()
sut.search_hyperparms(modelAdapter, hyperparams_arg)
