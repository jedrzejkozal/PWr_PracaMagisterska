from geometry.points_operations import *


def move_object_from_to(point_begin, point_end, num_frames=20):
    #returns coordinates that should be used to print an image
    points = [point_begin]
    shift_vec = calc_shift_vector(point_begin, point_end, num_frames)
    return generate_trajectory(points, shift_vec, num_frames)


def calc_shift_vector(point_begin, point_end, num_frames):
    shift_x = (point_end[0] - point_begin[0]) / num_frames
    shift_y = (point_end[1] - point_begin[1]) / num_frames
    return (shift_x, shift_y)


def generate_trajectory(points, shift_vec, num_frames):
    current_position = points[0]
    for i in range(num_frames-1):
        current_position = move_point_by_vector(current_position, shift_vec)
        points.append(current_position)
    return points


def constant_position(point, num_frames=20):
    return move_object_from_to(point, point, num_frames=num_frames)


#def move_object_on_arc(point_begin, point_end, num_frames=20, clockwise=True):
