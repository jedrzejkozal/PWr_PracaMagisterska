from Tests.ModifiedReNetTestBase import *


class TestModifiedReNetDroupout(ModifiedReNetTestBase):

    @classmethod
    def setup_model_params(cls):
        cls.patch_size = 4
        cls.reNet_hidden_size = 3
        cls.fully_conn_hidden_size = 5
        cls.use_dropout = True
        cls.dropout_rate = 0.2


    @classmethod
    def setup(cls):
        cls.num_samples = 10
        cls.img_width = 8
        cls.img_height = 8
        cls.number_of_channels = 3
        cls.num_classes = 2
        cls.setup_model_params()
