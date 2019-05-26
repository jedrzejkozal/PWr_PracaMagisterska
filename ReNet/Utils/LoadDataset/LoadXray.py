from ReNet.Utils.LoadDataset.LoadDataset import *


class LoadXray(LoadDataset):

    def load(self):
        img_size = (64, 64)
        path = 'datasets/chest_xray/chest_xray/'
        x_train_data, y_train_data = self.load_data_from_directory(path+"train", 5216, img_size, color_mode="grayscale")
        x_test_data, y_test_data = self.load_data_from_directory(path+"test", 624, img_size, color_mode="grayscale")
        x_val_data, y_val_data = self.load_data_from_directory(path+"val", 16, img_size, color_mode="grayscale")
        print("before undersampling: ", x_train_data.shape, y_train_data.shape)

        x_train_data = self.normalize(x_train_data)
        x_test_data = self.normalize(x_test_data)
        x_val_data = self.normalize(x_val_data)

        return x_train_data, y_train_data
