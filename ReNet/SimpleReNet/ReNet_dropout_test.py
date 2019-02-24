from ReNetTest import *
from SimpleReNet import *


class TestDropoutReNet(ReNetTest):

    @classmethod
    def setup_model_params(cls):
        cls.w_p = 2
        cls.h_p = 2
        cls.I = cls.img_width // cls.w_p
        cls.J = cls.img_height // cls.h_p
        cls.reNet_hidden_size = 1
        cls.fully_conn_hidden_size = 1
        cls.num_classes = 2
        cls.reNet_dropout = 0.2
        cls.fully_conn_dropout = 0.2


    @classmethod
    def setup(cls):
        cls.num_samples = 60
        cls.img_width = 10
        cls.img_height = 10
        cls.number_of_channels = 1
        cls.setup_model_params()
        cls.SUT_class = SimpleReNet


    @pytest.fixture
    def sut(self):
        self.__class__.setup()
        model = self.SUT_class([[self.w_p, self.h_p]],
                self.reNet_hidden_size, self.fully_conn_hidden_size, self.num_classes,
                use_dropout=True, reNet_dropout=self.reNet_dropout, fully_conn_dropout=self.fully_conn_dropout)
        model.compile(loss='categorical_crossentropy', optimizer='adam',
                metrics=['categorical_accuracy'])
        return model
