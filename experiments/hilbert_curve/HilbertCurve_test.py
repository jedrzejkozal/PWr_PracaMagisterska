import pytest

from HilbertCurve import *


class TestHilbertCurve(object):

    @pytest.fixture
    def sut(self):
        return HilbertCurve()

    def test_img_4x5_ValueError_raised(self, sut):
        arg = np.zeros((4,5))

        with pytest.raises(ValueError) as e:
            result = sut.hilbert_curve(arg)
            assert result == None

        assert "Invalid img shape" in str(e.value)


    def test_img_5x5_ValueError_raised(self, sut):
        arg = np.zeros((5,5))

        with pytest.raises(ValueError) as e:
            result = sut.hilbert_curve(arg)
            assert result == None

        assert "Invalid img shape" in str(e.value)


    def test_img_4x4_is_converted_to_16x1(self, sut):
        arg = np.zeros((4, 4))

        result = sut.hilbert_curve(arg)
        assert result.shape == (16, 1)


    def test_generate_commands_for_deegre_1_generates_A(self, sut):
        result = sut.generate_commands(1)
        assert result == ["left", "F", "right", "F", "right", "F", "left"]


    def test_generate_commands_for_deegre_2_generates_right_sequence(self, sut):
        result = sut.generate_commands(2)
        expected = ["left", "right", "F", "left", "F", "left", "F", "right", "F", "right",
                    "left", "F", "right", "F", "right", "F", "left", "F",
                    "left", "F", "right", "F", "right", "F", "left", "right", "F",
                    "right", "F", "left", "F", "left", "F", "right", "left"]

        assert result == expected


    def test_all_items_of_img_4x4_are_converted_properly(self, sut):
        arg = np.array(list(range(16)))
        arg = arg.reshape((4, 4))

        result = sut.hilbert_curve(arg)
        expected = np.array([12, 13, 9, 8,
                            4, 0, 1, 5,
                            6, 2, 3, 7,
                            11, 10, 14, 15])
        expected = expected.reshape((16, 1))

        assert np.array_equal(result, expected)


    def test_all_items_of_img_8x8_are_converted_properly(self, sut):
        arg = np.array(list(range(64)))
        arg = arg.reshape((8, 8))

        result = sut.hilbert_curve(arg)
        expected = np.array([56, 48, 49, 57, 58, 59, 51, 50,
                            42, 43, 35, 34, 33, 41, 40, 32,
                            24, 25, 17, 16, 8, 0, 1, 9,
                            10, 2, 3, 11, 19, 18, 26, 27,
                            28, 29, 21, 20, 12, 4, 5, 13,
                            14, 6, 7, 15, 23, 22, 30, 31,
                            39, 47, 46, 38, 37, 36, 44, 45,
                            53, 52, 60, 61, 62, 54, 55, 63])
        expected = expected.reshape((64, 1))

        assert np.array_equal(result, expected)
