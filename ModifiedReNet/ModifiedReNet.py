from keras import Model

from HilbertLayer import *
from ModifiedReNetLayer import *


class ModifiedReNet(Model):

    def __init__(self, patch_size, reNet_hidden_size, fully_conn_hidden_size):
        super().__init__()


    def call(self, inputs):
        pass
