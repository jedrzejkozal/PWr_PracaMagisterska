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

        line_begin0, line_end0, line_step0 = self.get_line_for_curve(h0, 4**2 - 1)
        line_begin1, line_end1, line_step1 = self.get_line_for_curve(h1, 4**3 - 1)
        line_begin2, line_end2, line_step2 = self.get_line_for_curve(h2, 4**4 - 1)
        line_begin3, line_end3, line_step3 = self.get_line_for_curve(h3, 4**5 - 1)

        line0 = Line(line_begin0, line_end0, color=BLACK)
        line1 = Line(line_begin1, line_end1, color=BLACK)
        line2 = Line(line_begin2, line_end2, color=BLACK)
        line3 = Line(line_begin3, line_end3, color=BLACK)

        animations = self.simultaneous_animations([h0, h1, h2, h3], ShowIncreasingSubsets)
        animations = animations + self.simultaneous_animations([line0, line1, line2, line3], ShowIncreasingSubsets)
        self.play(*animations)

        self.create_dots_and_traverse(h0, line_begin0, line_step0, 4**2 - 1, mark_at=[2, 7, 10, 99999999])
        self.create_dots_and_traverse(h1, line_begin1, line_step1, 4**3 - 1, mark_at=[10, 31, 42, 99999999])
        #self.create_dots_and_traverse(h2, line_begin2, line_step2, 4**4 - 1, mark_at=[37, 99999999])
        #self.create_dots_and_traverse(h3, line_begin3, line_step3, 4**5 - 1, mark_at=[])


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


    def get_line_for_curve(self, curve, number_of_sections):
        line_begin = np.array([curve.get_x(), curve.get_y(), 0]) + np.array([4,0,0])
        line_end = line_begin + np.array([5,0,0])
        line_step = (line_end - line_begin) / number_of_sections
        return line_begin, line_end, line_step


    def simultaneous_animations(self, objects_list, animation):
        return tuple([animation(object) for object in objects_list])


    def create_dots_and_traverse(self, curve, line_begin, line_step, number_of_sections, mark_at=[]):
        dot_curve, curve_movement = self.get_curve_dot_and_movement(curve)
        dot_line, line_movement = self.get_line_dot_and_movement(line_begin, line_step, number_of_sections)
        self.play(ShowCreation(dot_curve), ShowCreation(dot_line))
        self.traverse(dot_curve, curve_movement, dot_line, line_movement, runtime=1.0/number_of_sections, mark_at=mark_at)
        self.play(Uncreate(dot_curve), Uncreate(dot_line))


    def get_curve_dot_and_movement(self, curve):
        curve_begin = curve.get_points()[-1]
        dot_curve = Dot(curve_begin, color=RED)
        curve_movement = self.points_to_vectors(curve.get_points())
        return dot_curve, curve_movement


    def get_line_dot_and_movement(self, line_begin, line_step, number_of_sections):
        dot_line = Dot(line_begin, color=RED)
        line_movement = [line_step] * number_of_sections
        return dot_line, line_movement


    def traverse(self, dot_curve, curve_movement, dot_line, line_movement, runtime=1, mark_at=[]):
        i = 0
        for vec_curve, vec_line in zip(curve_movement, line_movement):
            if i == mark_at[0]:
                mark_at.pop(0)
                self.add(dot_curve.deepcopy())
                self.add(dot_line.deepcopy())
            p1 = PhaseFlow(lambda x: vec_curve, dot_curve)
            p2 = PhaseFlow(lambda x: vec_line, dot_line)
            p1.set_run_time(runtime)
            p1.set_run_time(runtime)
            self.play(p1, p2)
            i += 1


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
