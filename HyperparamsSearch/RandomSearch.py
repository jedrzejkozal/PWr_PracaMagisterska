from ModelIfc import *


class RandomSearch(object):

    def __init__(self, num_try=10):
        pass


    def search_hyperparms(self, model, hyperparams_boundries):
        self.__check_model_type(model)


    def __check_model_type(self, model):
        if type(model.__class__.__bases__[0]) is not type(ModelIfc):
            raise TypeError("invalid model type")
