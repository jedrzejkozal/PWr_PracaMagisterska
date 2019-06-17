from PIL import ImageDraw
from math import pi

from objects.Object import *
from geometry.points_operations import *


class Line(Object):

    def __init__(self, point_begin, point_end, *args, width=0, has_arrow=False, **kwargs):
        super().__init__(*args, **kwargs)

        self.point_begin = point_begin
        self.point_end = point_end
        self.width = width
        self.has_arrow = has_arrow

        self.__calc_vectors()
        if self.has_arrow:
            self.calc_arrow_arms_position()


    def calc_arrow_arms_position(self):
        direction = normalize_vector(self.vec_to_end)
        vec = multiply_vector_by_scalar(direction, 5)
        point = move_point_by_vector(self.point_end, vec)
        point_in_origin_reference_system = move_point_by_vector(point, opposite_vector(self.point_end))
        self.arrow_left_arm_vec = rotate_point_by_theta(point_in_origin_reference_system, 5.0/6.0*pi)
        self.arrow_right_arm_vec = rotate_point_by_theta(point_in_origin_reference_system, -5.0/6.0*pi)


    def __calc_vectors(self):
        self.middle = get_middle(self.point_begin, self.point_end)
        self.vec_to_end = get_vector(self.point_begin, self.point_end)


    def draw(self, img, point):
        draw = ImageDraw.Draw(img)
        line_end = move_point_by_vector(point, self.vec_to_end)
        draw.line((point, line_end), fill=self.color, width=self.width)
        if self.has_arrow:
            self.calc_arrow_arms_position()
            arrow_left_arm = move_point_by_vector(line_end, self.arrow_left_arm_vec)
            arrow_right_arm = move_point_by_vector(line_end, self.arrow_right_arm_vec)
            draw.line((arrow_left_arm, line_end), fill=self.color, width=self.width)
            draw.line((arrow_right_arm, line_end), fill=self.color, width=self.width)
        return img


    def rotation(self, theta):
        self.point_begin = rotate_point_by_theta(self.point_begin, theta)
        self.point_end = rotate_point_by_theta(self.point_end, theta)
        self.__calc_vectors()


    def extend(self, amount):
        direction = normalize_vector(self.vec_to_end)
        extend_vec = multiply_vector_by_scalar(direction, amount)
        self.point_end = move_point_by_vector(self.point_end, extend_vec)

        self.__calc_vectors()


    def show_arrow(self):
        self.has_arrow = True
        self.calc_arrow_arms_position()
