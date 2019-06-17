import sys
sys.path.insert(0,'/home/jkozal/Dokumenty/PWr/magisterka/magisterka/ReadmeFiles/manimlib/')
from HilbertCurve import *

from manimlib.imports import *


class HilbertTraversing(Scene):

    def setup(self):
        self.camera.set_frame_width(self.camera.get_frame_width()+1)
        self.camera.set_frame_height(self.camera.get_frame_height()+6)

        self.camera.reset_pixel_shape(2000, 1440)


    def construct(self):
        base_position = np.array([-4,8,0])
        vec_to_next_curve = np.array([0,-5, 0])
        h0 = self.get_curve_at(4, base_position)
        h1 = self.get_curve_at(8, base_position+1*vec_to_next_curve)
        h2 = self.get_curve_at(16, base_position+2*vec_to_next_curve)
        h3 = self.get_curve_at(32, base_position+3*vec_to_next_curve)

        animations = self.simultaneous_animations([h0, h1, h2, h3], ShowIncreasingSubsets)
        self.play(*animations)
        self.traversing_point(h0, 2)
        self.traversing_point(h1, 3)
        #self.traversing_point(h2, 4)
        #self.traversing_point(h3, 5)


    def traversing_point(self, curve, degree):
        curve_begin = curve.get_points()[-1]
        dot_curve = Dot(curve_begin, color=RED)
        self.play(ShowCreation(dot_curve))

        curve_movement = self.points_to_vectors(curve.get_points())
        line_begin = np.array([curve.get_x(), curve.get_y(), 0]) + np.array([4,0,0])

        number_of_sections = 4**degree - 1
        line_end = line_begin + np.array([5,0,0])
        self.play(ShowIncreasingSubsets(Line(line_begin, line_end, color=BLACK)))
        line_step = (line_end - line_begin) / number_of_sections

        dot_line = Dot(line_begin, color=RED)
        self.play(ShowCreation(dot_line))


        line_movement = [line_step] * number_of_sections
        self.traverse(dot_curve, curve_movement, dot_line, line_movement)


    def get_curve_at(self, curve_side_length, point):
        h1 = VMobject(color=BLACK)
        points = self.get_hilbert_curve(curve_side_length)
        h1.set_points(points)
        h1.set_points_as_corners(points)
        h1.set_x(point[0])
        h1.set_y(point[1])
        h1.rotate(PI/2)
        return h1


    def get_hilbert_curve(self, curve_side_length):
        indexes = self.get_image_indexes(curve_side_length)
        hilbert_curve = HilbertCurve1()
        points = hilbert_curve.convert_2D_to_1D(indexes)
        points =  self.rescale_to_the_middle(points, curve_side_length)
        points = self.rescale_to_curve_side_length(points, 2, curve_side_length)
        points = list(map(lambda x: np.append(x, [0]), points))
        return points


    def get_image_indexes(self, curve_side_length):
        result = []

        for i in reversed(range(curve_side_length)):
            for j in range(curve_side_length):
                result.append([i, j])

        result_matrix = np.array(result)
        return result_matrix.reshape((curve_side_length, curve_side_length, 2))


    def rescale_to_the_middle(self, points, curve_side_length):
        points = points + np.array([-curve_side_length/4,-curve_side_length/4])
        return points * 2


    def rescale_to_curve_side_length(self, points, new_side_length, old_side_length):
        return points * (new_side_length/old_side_length)


    def simultaneous_animations(self, objects_list, animation):
        return tuple([animation(object) for object in objects_list])


    def traverse(self, dot_curve, curve_movement, dot_line, line_movement):
        for vec_curve, vec_line in zip(curve_movement, line_movement):
            self.play(PhaseFlow(lambda x: vec_curve, dot_curve),
                        PhaseFlow(lambda x: vec_line, dot_line))


    def points_to_vectors(self, points):
        vectors = [points[len(points)-2] - points[len(points)-1]]
        self.vec_with_same_direction_count = 0
        for i in range(len(points)-3, -1, -1):
            vec = points[i] - points[i+1]
            self.add_non_zero_vec(vectors, vec)
        return vectors


    def add_non_zero_vec(self, vectors, vec):
        if not self.is_zero_vector(vec):
            if self.vec_with_same_direction_count < 2:
                vectors[-1] = vectors[-1] + vec
                self.vec_with_same_direction_count += 1
            else:
                vectors.append(vec)
                self.vec_with_same_direction_count = 0


    def is_zero_vector(self, vec):
        return np.isclose(vec, np.array([0,0,0])).all()


    def directions_are_the_same(self, vec1, vec2):
        return np.isclose(vec1/np.linalg.norm(vec1), vec2/np.linalg.norm(vec2)).all()
