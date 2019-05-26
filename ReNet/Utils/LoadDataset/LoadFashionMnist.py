from keras.utils import to_categorical
from keras.datasets import fashion_mnist
from PIL import Image

from ReNet.Utils.LoadDataset.LoadDataset import *


class LoadFashionMnist(LoadDataset):

    def load(self):
        img_rows, img_cols = 28, 28
        num_classes = 10
        (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        x_train_data = self.normalize(x_train)
        x_test_data = self.normalize(x_test)

        # convert class vectors to binary class matrices
        y_train_data = to_categorical(y_train, num_classes)
        y_test_data = to_categorical(y_test, num_classes)

        new_size = (32, 32)
        x_train_data = self.resize_data(x_train_data, new_size)
        x_test_data = self.resize_data(x_test_data, new_size)
        print(x_train_data.shape)
        print(x_test_data.shape)

        return x_train_data, y_train_data


    def resize_data(self, data, new_size):
        num_samples = data.shape[0]
        resized = np.zeros((num_samples,)+new_size+(1,), dtype=data.dtype)

        for i in range(num_samples):
            img = Image.fromarray(np.squeeze(data[i]))
            resized_img = np.asarray(img.resize(new_size))
            resized[i] = np.expand_dims(resized_img, axis=3)

        return resized
