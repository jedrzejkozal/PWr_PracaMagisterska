import pytest
from math import pi
from geometry.points_operations import *

def test_rotate_point_2_1_by_90_deg_relative_to_3_1_result_is_2_2():
    origin = (2,1)
    point = (3,1)
    result = rotate_point_by_theta_relative_to_origin(point, pi/2.0, origin)
    assert result == (2,2)


def test_rotate_point_2_1_by_neg_90_deg_relative_to_3_1_result_is_2_0():
    origin = (2,1)
    point = (3,1)
    result = rotate_point_by_theta_relative_to_origin(point, pi/-2.0, origin)
    assert result == (2,0)
