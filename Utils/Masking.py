import numpy as np

class Masking(object):

    def __init__(self, img_rows, img_cols, num_channels):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.num_channels = num_channels


    def mask_input(self, x):
        print("mask_input call")
        mask = self.__generate_mask_with_prob(0.2)
        return self.__apply_mask(x, mask)


    def __generate_mask_with_prob(self, p):
        rand_matrix = self.__get_random_binomial(p, self.__get_patched_img_size())
        expand_matrix = self.__get_expand_matrix(self.img_rows // 2)
        single_dim_mask = np.transpose(expand_matrix) @ rand_matrix @ expand_matrix

        all_dims = [single_dim_mask] * self.num_channels
        mask = np.stack(all_dims, axis=2)

        return mask > 0.5


    def __get_patched_img_size(self):
        img_size = self.img_rows * self.img_cols
        return img_size // 4  # patch_size_x*patch_size_y = 2*2 = 4


    def __get_random_binomial(self, p, patched_img_size):
        rand_matrix = np.random.binomial(1, p, size=patched_img_size)
        return rand_matrix.reshape(self.img_rows // 2, self.img_cols // 2)


    def __get_expand_matrix(self, n):
        expand_matrix = np.zeros((n, 2*n)) #won't work for rectangular imgs
        for i in range(n):
            expand_matrix[i, 2*i:2*i+2] = 1
        return expand_matrix


    def __apply_mask(self, x, mask):
        tmp = np.copy(x)
        for i in range(x.shape[0]):
            tmp[i][mask] = -100.0
        return tmp
