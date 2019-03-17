#code from blog:
#http://www.pinchofintelligence.com/simple-introduction-to-tensorboard-embedding-visualisation/
import matplotlib.pyplot as plt
import numpy as np

import os


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))


    spriteimage = np.ones((img_h * n_plots ,img_w * n_plots ))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                  j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits,(-1,28,28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1-mnist_digits


def save_sprites(x_test, log_dir):
    to_visualise = vector_to_matrix_mnist(x_test)
    to_visualise = invert_grayscale(to_visualise)
    sprite_image = create_sprite_image(to_visualise)

    path_for_mnist_sprites =  os.path.join(log_dir, 'sprite.png')
    plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
    plt.imshow(sprite_image, cmap='gray')
