import keras
from keras_preprocessing.image import NumpyArrayIterator
from keras_preprocessing.image import ImageDataGenerator

import numpy as np



class NumpyArrayIteratorWithMasking(NumpyArrayIterator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orginal_x = np.copy(self.x)
        self.mask_input()


    def mask_input(self):
        print("mask_input call")
        mask = self.generate_mask_with_prob(0.2)
        print(np.bincount(mask.reshape(mask.size)))

        tmp = np.copy(self.orginal_x)
        for i in range(self.x.shape[0]):
            tmp[i][mask] = -100.0
        self.x = tmp


    def generate_mask_with_prob(self, p):
        rand = np.random.binomial(1, p, size=self.x.shape[1]*self.x.shape[2]*self.x.shape[3])
        mask = rand.reshape(self.x.shape[1], self.x.shape[2], self.x.shape[3])
        return mask > 0.5


    def on_epoch_end(self):
        super().on_epoch_end()
        self.mask_input()



class ImageDataGeneratorWithMasking(ImageDataGenerator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    def flow(self, x,
             y=None, batch_size=32, shuffle=True,
             sample_weight=None, seed=None,
             save_to_dir=None, save_prefix='', save_format='png', subset=None):
        return NumpyArrayIteratorWithMasking(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            sample_weight=sample_weight,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            subset=subset)



class ImageDataGenerator(ImageDataGeneratorWithMasking):

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 validation_split=0.0,
                 dtype=None):
        if data_format is None:
            data_format = backend.image_data_format()
        kwargs = {}
        if 'dtype' in inspect.getargspec(
                image.ImageDataGenerator.__init__).args:
            if dtype is None:
                dtype = backend.floatx()
            kwargs['dtype'] = dtype
        super(ImageDataGenerator, self).__init__(
            featurewise_center=featurewise_center,
            samplewise_center=samplewise_center,
            featurewise_std_normalization=featurewise_std_normalization,
            samplewise_std_normalization=samplewise_std_normalization,
            zca_whitening=zca_whitening,
            zca_epsilon=zca_epsilon,
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            brightness_range=brightness_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            fill_mode=fill_mode,
            cval=cval,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
            rescale=rescale,
            preprocessing_function=preprocessing_function,
            data_format=data_format,
            validation_split=validation_split,
            **kwargs)
