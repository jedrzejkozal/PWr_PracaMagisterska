import pytest
import numpy as np

from geometry.translation import *

def test_move_object_from_0_0_to_1_1_all_points_are_in_line():
    points = move_object_from_to((0,0),(1,1))

    for p in points:
        assert np.isclose(p[0], p[1])


def test_move_object_from_to_number_of_generated_points_is_equal_to_num_frames():
    points = move_object_from_to((0,0),(1,1), num_frames=10)
    assert len(points) == 10
