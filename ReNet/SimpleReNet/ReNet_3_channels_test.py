from ReNetTest import *
from SimpleReNet import *


class Test3ChannelsReNet(ReNetTest):

    @classmethod
    def setup_model_params(cls):
        cls.w_p = 2
        cls.h_p = 2
        cls.I = cls.img_width // cls.w_p
        cls.J = cls.img_height // cls.h_p
        cls.reNet_hidden_size = 1
        cls.fully_conn_hidden_size = 1
        cls.num_classes = 2


    @classmethod
    def setup(cls):
        cls.num_samples = 60
        cls.img_width = 12
        cls.img_height = 12
        cls.number_of_channels = 3
        cls.setup_model_params()
