from math import cos, sin
from math import sqrt

def move_point_by_vector(point, vec):
    return (point[0] + vec[0], point[1] + vec[1])


def get_middle(p1, p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)


def get_vector(begin, end):
    return (end[0]-begin[0], end[1]-begin[1])


def rotate_point_by_theta_relative_to_origin(point, theta, origin):
    point_in_origin_reference_system = move_point_by_vector(point, opposite_vector(origin))
    rotated = rotate_point_by_theta(point_in_origin_reference_system, theta)
    return move_point_by_vector(rotated, origin)


def rotate_point_by_theta(point, theta): #theta is in radians
    rotation_matrix = [[cos(theta), -1.0*sin(theta)],[sin(theta), cos(theta)]]
    return multiply_matix_by_point(rotation_matrix, point)


def multiply_matix_by_point(matrix, point):
    return (matrix[0][0]*point[0] + matrix[0][1]*point[1],matrix[1][0]*point[0] + matrix[1][1]*point[1])


def vector_length(vec):
    return sqrt(vec[0]*vec[0] + vec[1]*vec[1])


def normalize_vector(vec):
    vec_len = vector_length(vec)
    return (vec[0] / vec_len, vec[1] / vec_len)


def multiply_vector_by_scalar(vec, scalar):
    return (vec[0]*scalar, vec[1]*scalar)


def opposite_vector(vec):
    return (-vec[0], -vec[1])
