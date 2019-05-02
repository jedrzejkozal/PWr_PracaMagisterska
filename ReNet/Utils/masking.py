import numpy as np

def mask_input(x):
    print("mask_input call")
    mask = generate_mask_with_prob(0.2)

    tmp = np.copy(x)
    for i in range(x.shape[0]):
        tmp[i][mask] = -100.0
    return tmp

def generate_mask_with_prob(p):
    rand_matrix = get_random_binomial(p, get_patched_img_size())
    expand_matrix = get_expand_matrix(img_rows // 2)
    single_dim_mask = np.transpose(expand_matrix) @ rand_matrix @ expand_matrix

    all_dims = [single_dim_mask]
    mask = np.stack(all_dims, axis=2)

    return mask > 0.5


def get_patched_img_size():
    img_size = img_rows * img_cols
    return img_size // 4  # patch_size_x*patch_size_y = 2*2 = 4


def get_random_binomial(p, patched_img_size):
    rand_matrix = np.random.binomial(1, p, size=patched_img_size)
    return rand_matrix.reshape(img_rows // 2, img_cols // 2)


def get_expand_matrix(n):
    expand_matrix = np.zeros((n, 2*n)) #won't work for rectangular imgs
    for i in range(n):
        expand_matrix[i, 2*i:2*i+2] = 1
    return expand_matrix
