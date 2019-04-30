from keras import backend as K

from Models.ReNetLayer import *


class IdentityReNetLayer(ReNetLayer):

    def __init__(self, size_of_patches, hidden_size,
                    use_dropout=False, dropout_rate=None,
                    is_first_layer=False):
        super().__init__(size_of_patches, hidden_size,
                    use_dropout=use_dropout, dropout_rate=dropout_rate,
                    is_first_layer=is_first_layer)

        def constant_activation(x):
            return K.constant(1, shape=[hidden_size])


        self.LSTM_up_down = LSTM(hidden_size,
                        activation=constant_activation,
                        recurrent_activation='linear',
                        return_sequences=True)
        self.LSTM_down_up = LSTM(hidden_size,
                        activation=constant_activation,
                        recurrent_activation='linear',
                        return_sequences=True,
                        go_backwards=True)
        self.LSTM_left_right = LSTM(hidden_size,
                        activation=constant_activation,
                        recurrent_activation='linear',
                        return_sequences=True)
        self.LSTM_right_left = LSTM(hidden_size,
                        activation=constant_activation,
                        recurrent_activation='linear',
                        return_sequences=True,
                        go_backwards=True)
