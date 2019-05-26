from abc import ABC, abstractmethod
import numpy as np

from keras.preprocessing.image import ImageDataGenerator


class LoadDataset(ABC):

    def load_data_from_directory(self, dir, num_samples, size, color_mode='rgb'):
        datagen = ImageDataGenerator()
        iterator = datagen.flow_from_directory(dir, batch_size=num_samples, target_size=size, color_mode=color_mode)
        return next(iterator)

    def normalize(self, matrix):
        mu = np.mean(matrix, axis=0)
        std = np.std(matrix, axis=0)
        return (matrix - mu) / std

    @abstractmethod
    def load(self):
        pass
