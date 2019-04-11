from abc import ABC
from abc import abstractmethod


class ModelIfc(ABC):

    @abstractmethod
    def train(self, *hyperparams):
        pass


    @abstractmethod
    def evaluate(self, *hyperparams):
        pass
