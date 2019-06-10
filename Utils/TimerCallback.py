import time
import numpy as np

from keras.callbacks import Callback

class TimerCallback(Callback):

    def __init__(self):
        self.measurements = []


    def on_train_begin(self, logs):
        self.measurements = []


    def on_epoch_begin(self, epoch, logs):
        self.start_time = time.time()


    def on_epoch_end(self, epoch, logs):
        epoch_duration = time.time() - self.start_time
        self.measurements.append(epoch_duration)


    def get_results(self):
        return np.array(self.measurements)
