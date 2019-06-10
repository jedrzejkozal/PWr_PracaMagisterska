from ReNet.Utils.LoadDataset.LoadDataset import *

class LoadFlowers(LoadDataset):

    def load(self):
        num_samples = 4323
        x_data, y_data = self.load_data_from_directory("datasets/flowers-recognition/flowers", num_samples, (32, 32))
        print("before undersampling: ", x_data.shape, y_data.shape)

        x_data = self.normalize(x_data)

        return x_data, y_data
