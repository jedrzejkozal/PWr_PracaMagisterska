import pytest
import numpy as np

from Utils.ImageGeneratorWithMasking import *

class TestImageGeneratorWithMasking(object):

    def test_generate_mask_one_channel(self):
        x = np.zeros((10, 6, 6, 1))
        sut = NumpyArrayIteratorWithMasking(x, x, x)
        res = sut.generate_mask_with_prob(0.2)
        assert res.shape == (6, 6, 1)


    def test_generate_mask_three_channel(self):
        x = np.zeros((10, 6, 6, 3))
        sut = NumpyArrayIteratorWithMasking(x, x, x)
        res = sut.generate_mask_with_prob(0.2)
        assert res.shape == (6, 6, 3)


    def test_mask_input(self):
        x = np.zeros((10, 6, 6, 1))
        sut = NumpyArrayIteratorWithMasking(x, x, x)
        sut.mask_input()
        assert sut.x.shape == (10, 6, 6, 1)
