from Tests.ReNetLayerTest import *


class TestHorizontalPatchLayer(ReNetLayerTest):

    @classmethod
    def setup_model_params(cls):
        cls.w_p = 4
        cls.h_p = 3
        cls.I = cls.img_width // cls.w_p
        cls.J = cls.img_height // cls.h_p
        cls.hidden_size = 1


    @classmethod
    def setup(cls):
        cls.num_samples = 60
        cls.img_width = 12
        cls.img_height = 12
        cls.number_of_channels = 1
        cls.setup_model_params()
