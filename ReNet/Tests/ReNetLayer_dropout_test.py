from Tests.ReNetLayerTest import *


class TestDropoutReNetLayer(ReNetLayerTest):

    @classmethod
    def setup_model_params(cls):
        cls.w_p = 2
        cls.h_p = 2
        cls.I = cls.img_width // cls.w_p
        cls.J = cls.img_height // cls.h_p
        cls.hidden_size = 1
        cls.dropout = 0.2


    @classmethod
    def setup(cls):
        cls.num_samples = 60
        cls.img_width = 10
        cls.img_height = 10
        cls.number_of_channels = 1
        cls.setup_model_params()


    @pytest.fixture
    def sut(self):
        self.__class__.setup()

        return ReNetLayer([[self.w_p, self.h_p]], self.hidden_size,
                    use_dropout=True, dropout_rate=self.dropout)
