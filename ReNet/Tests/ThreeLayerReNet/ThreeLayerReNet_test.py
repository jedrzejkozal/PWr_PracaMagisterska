from Tests.ReNetTestBase import *
from Models.ThreeLayerReNet.ThreeLayerReNet import *


class TestThreeLayerReNet(ReNetTestBase):

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
        cls.img_width = 24
        cls.img_height = 24
        cls.number_of_channels = 1
        cls.setup_model_params()
        cls.SUT_class = ThreeLayerReNet
