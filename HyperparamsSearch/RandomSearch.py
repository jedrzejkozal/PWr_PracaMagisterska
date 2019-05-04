from ModelIfc import *


class RandomSearch(object):

    def __init__(self, num_try=10):
        self.num_try = num_try


    def search_hyperparms(self, model, hyperparams_generators):
        self.__check_model_type(model)
        for i in range(self.num_try):
            hyperparams = self.__generate_hyperparams(hyperparams_generators)
            print("{}/{}\t{}".format(i+1, self.num_try,
                    self.__formated_hyperparams(hyperparams)))
            self.__train_and_evaluate(model, hyperparams)


    def __check_model_type(self, model):
        if type(model.__class__.__bases__[0]) is not type(ModelIfc):
            raise TypeError("invalid model type")


    def __generate_hyperparams(self, hyperparams_generators):
        hyperparams = {}
        for name, generator in hyperparams_generators.items():
            hyperparams[name] = generator()

        return hyperparams


    def __formated_hyperparams(self, hyperparams):
        result = ""
        for name, val in hyperparams.items():
            result += "{}={} ".format(name, val)
        return result


    def __train_and_evaluate(self, model, hyperparams_list):
        model.train(hyperparams_list)
        return model.evaluate(hyperparams_list)
