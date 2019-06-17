import sys
sys.path.insert(0,'/home/jkozal/Dokumenty/PWr/magisterka/magisterka/ReadmeFiles/manimlib/')
from HilbertCurve import *

from manimlib.imports import *


class Hilbert(Scene):

    def setup(self):
        self.camera.set_frame_width(self.camera.get_frame_width()+46)
        self.camera.set_frame_height(self.camera.get_frame_height()+25)

    def construct(self):
        base_position = np.array([-26,-14,0])
        h0 = self.get_curve_at(2, base_position)
        side_length = 0

        max_deg = 3
        for degree in range(1,max_deg+1):
            side_length = side_length*2 + 2 #2*(2*degree-1)
            move_to_next_deegre = np.array([side_length+1,0,0])
            move_up = np.array([0,side_length+2,0])
            move_right = np.array([side_length+2,0,0])
            move_down = np.array([0,-1*side_length-2,0])

            h1 = self.get_curve_at(2**degree, base_position)
            position_1st = base_position + move_to_next_deegre
            h2 = self.get_curve_at(2**degree, position_1st)
            position_2nd = position_1st + move_up
            h3 = self.get_curve_at(2**degree, position_2nd)
            position_3rd = position_2nd + move_right
            h4 = self.get_curve_at(2**degree, position_3rd)
            position_4th = position_3rd + move_down
            connect1_2 = Line(position_1st+np.array([side_length/-2,side_length/2,0]), position_2nd+np.array([side_length/-2,side_length/-2,0]), color=BLACK)
            connect2_3 = Line(position_2nd+np.array([side_length/2,side_length/-2,0]), position_3rd+np.array([side_length/-2,side_length/-2,0]), color=BLACK)
            conncet3_4 = Line(position_3rd+np.array([side_length/2,side_length/-2,0]), position_4th+np.array([side_length/2,side_length/2,0]), color=BLACK)


            self.play(ShowIncreasingSubsets(h0), ShowIncreasingSubsets(h1))
            self.play(PhaseFlow(lambda x: move_to_next_deegre, h1))
            self.play(ShowIncreasingSubsets(h2), PhaseFlow(lambda x: move_up, h2))
            self.play(ShowIncreasingSubsets(h3), PhaseFlow(lambda x: move_right, h3))
            self.play(ShowIncreasingSubsets(h4), PhaseFlow(lambda x: move_down, h4))
            self.play(Rotate(h1, angle=PI/-2.0), Rotate(h4, angle=PI/2.0))
            self.play(ShowCreation(connect1_2), ShowCreation(connect2_3), ShowCreation(conncet3_4))

            base_position = self.get_middle_point(position_1st, position_3rd)


    def get_curve_at(self, side_length, point):
        h1 = VMobject(color=BLACK)
        h1.set_points(self.get_hilbert_curve(side_length))
        h1.set_points_as_corners(self.get_hilbert_curve(side_length))
        h1.set_x(point[0])
        h1.set_y(point[1])
        h1.rotate(PI/2)
        return h1


    def get_hilbert_curve(self, side_length):
        indexes = self.get_indexes(side_length)
        hilbert_curve = HilbertCurve1()
        points = hilbert_curve.convert_2D_to_1D(indexes)
        points = points + np.array([-side_length/4,-side_length/4])
        points = points * 2
        points = list(map(lambda x: np.append(x, [0]), points))
        return points


    def get_indexes(self, side_length):
        result = []

        for i in reversed(range(side_length)):
            for j in range(side_length):
                result.append([i, j])

        result_matrix = np.array(result)
        return result_matrix.reshape((side_length, side_length, 2))


    def get_middle_point(self, p1, p2):
        return np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, 0])
