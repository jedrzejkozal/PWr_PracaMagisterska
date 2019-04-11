import pytest

from CartesianProduct import *


def test_cartesian_product_of_wrong_tuple_initialistation_wrong_type_raised():
    arg = ([1])

    with pytest.raises(TypeError) as err:
        result = cartesian_product(arg)
        assert result is None

    assert "recived arg is not tuple of lists" in str(err.value)


def test_cartesian_product_of_list_initialistation_wrong_type_raised():
    arg = [1]

    with pytest.raises(TypeError) as err:
        result = cartesian_product(arg)
        assert result is None

    assert "recived arg is not tuple of lists" in str(err.value)


def test_cartesian_product_of_1_list_with_2_elems():
    arg = ([1, 2],)

    result = cartesian_product(arg)

    assert result[0] == [1, 2]


def test_cartesian_product_of_2_list_with_1_and_2_elems():
    arg = ([1], [2, 3])

    result = cartesian_product(arg)

    assert result[0] == [1, 2]
    assert result[1] == [1, 3]


def test_cartesian_product_of_2_list_with_2_elems():
    arg = ([1, 2], [3, 4])

    result = cartesian_product(arg)

    assert result[0] == [1, 3]
    assert result[1] == [1, 4]
    assert result[2] == [2, 3]
    assert result[3] == [2, 4]


def test_cartesian_product_of_2_list_with_3_elems():
    arg = ([1, 2, 3], [4, 5, 6])

    result = cartesian_product(arg)

    assert result[0] == [1, 4]
    assert result[1] == [1, 5]
    assert result[2] == [1, 6]
    assert result[3] == [2, 4]
    assert result[4] == [2, 5]
    assert result[5] == [2, 6]
    assert result[6] == [3, 4]
    assert result[7] == [3, 5]
    assert result[8] == [3, 6]


def test_cartesian_product_of_3_list_with_2_elems():
    arg = ([1, 2], [3, 4], [5, 6])

    result = cartesian_product(arg)

    assert result[0] == [1, 3, 5]
    assert result[1] == [1, 3, 6]
    assert result[2] == [1, 4, 5]
    assert result[3] == [1, 4, 6]
    assert result[4] == [2, 3, 5]
    assert result[5] == [2, 3, 6]
    assert result[6] == [2, 4, 5]
    assert result[7] == [2, 4, 6]


def test_cartesian_product_of_2_list_with_2_and_5_elems():
    arg = ([1, 2], [3, 4, 5, 6, 7])

    result = cartesian_product(arg)

    assert result[0] == [1, 3]
    assert result[1] == [1, 4]
    assert result[2] == [1, 5]
    assert result[3] == [1, 6]
    assert result[4] == [1, 7]
    assert result[5] == [2, 3]
    assert result[6] == [2, 4]
    assert result[7] == [2, 5]
    assert result[8] == [2, 6]
    assert result[9] == [2, 7]
