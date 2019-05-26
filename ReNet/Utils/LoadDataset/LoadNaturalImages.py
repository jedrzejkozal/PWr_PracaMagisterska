from ReNet.Utils.LoadDataset.LoadDataset import *

class LoadNaturalImages(LoadDataset):

    def load(self):
        num_samples = 6899
        x_data, y_data = self.load_data_from_directory("datasets/natural-images/natural_images/", num_samples, (32, 32))

        print("before undersampling: ", x_data.shape, y_data.shape)

        x_data = self.normalize(x_data)

        return x_data, y_data
