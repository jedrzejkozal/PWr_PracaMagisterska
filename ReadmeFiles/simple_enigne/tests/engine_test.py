import pytest

from engine import *
from tests.objects.ObjectMock import *


def test_trajectories_contains_something_other_than_points_exception_raised():
    objects_list = [ObjectMock(0, 10)]
    trajectories = [[(0,0)], [(0,0), 1]]

    with pytest.raises(AssertionError) as err:
        generate_gif(objects_list, trajectories)

    assert err.type is AssertionError
    assert "trajectories must contain points saved as tuples" in str(err.value)


def test_trajectories_contains_points_have_3_coordinates_exception_raised():
    objects_list = [ObjectMock(0, 10)]
    trajectories = [[(0,0)], [(0,0), (1,0,2)]]

    with pytest.raises(AssertionError) as err:
        generate_gif(objects_list, trajectories)

    assert err.type is AssertionError
    assert "trajectories must contain 3 coordinates" in str(err.value)


def test_trajectories_number_is_not_equal_to_objects_number_exception_raised():
    objects_list = [ObjectMock(0, 10)]
    trajectories = [[(0,0), (0,0)], [(0,0), (1,0)]]

    with pytest.raises(AssertionError) as err:
        generate_gif(objects_list, trajectories)

    assert err.type is AssertionError
    assert "trajectories len must be equal to objects_list len" in str(err.value)


def test_trajectories_have_not_equal_number_of_points_expection_raised():
    objects_list = [ObjectMock(0, 10), ObjectMock(0, 10)]
    trajectories = [[(0,0), (0,0)], [(0,0), (1,0), (2,9)]]

    with pytest.raises(AssertionError) as err:
        generate_gif(objects_list, trajectories)

    assert err.type is AssertionError
    assert "trajectories len must be equal" in str(err.value)


def objects_list_contains_something_that_is_not_derived_from_Objects_class_exception_raised():
    objects_list = [ObjectMock(0, 10), 2]
    trajectories = [[(0,0), (0,0)], [(0,0), (1,0), (2,9)]]

    with pytest.raises(AssertionError) as err:
        generate_gif(objects_list, trajectories)

    assert err.type is AssertionError
    assert "objects must derive form Object class" in str(err.value)


def test_generate_gif_object_mock_drawn_frame_num_times():
    objects_list = [ObjectMock(0, 10)]
    trajectories = [[(0,0)]*10]

    generate_gif(objects_list, trajectories)
    assert objects_list[0].drawn == 10


def test_generate_gif_object_apears_when_its_specified():
    objects_list = [ObjectMock(3, 10)]
    trajectories = [[(0,0)]*10]

    generate_gif(objects_list, trajectories)
    assert objects_list[0].drawn == 7


def test_generate_gif_2_object_mock_drawn_frame_num_times():
    objects_list = [ObjectMock(0, 10), ObjectMock(0, 10)]
    trajectories = [[(0,0), (0,0), (0,0)], [(0,0), (1,0), (2,9)]]

    generate_gif(objects_list, trajectories)
    assert objects_list[0].drawn == 3
    assert objects_list[1].drawn == 3


def test_generate_gif_object_mock_apply_transformation_frame_num_times():
    objects_list = [ObjectMock(0, 10, transformations=[(ObjectMock.dummy_transformation, 0, -1)])]
    trajectories = [[(0,0)]*10]

    generate_gif(objects_list, trajectories)
    assert objects_list[0].transformations_applied == 10


def test_generate_gif_2_object_mock_apply_transformation_frame_num_times():
    objects_list = [ObjectMock(0, 10, transformations=[(ObjectMock.dummy_transformation, 0, -1)]),
                    ObjectMock(0, 10, transformations=[(ObjectMock.dummy_transformation, 0, -1)])]
    trajectories = [[(0,0), (0,0), (0,0)], [(0,0), (1,0), (2,9)]]

    generate_gif(objects_list, trajectories)
    assert objects_list[0].transformations_applied == 3
    assert objects_list[1].transformations_applied == 3


def test_generate_gif_object_mock_all_transformations_applyed():
    objects_list = [ObjectMock(0, 10, transformations=[(ObjectMock.dummy_transformation, 0, -1), (ObjectMock.dummy_transformation, 0, -1)])]
    trajectories = [[(0,0)]*10]

    generate_gif(objects_list, trajectories)
    assert objects_list[0].transformations_applied == 20
