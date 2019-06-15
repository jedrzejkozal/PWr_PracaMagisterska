from manimlib.imports import *


class Hilbert(Scene):

    def get_basic_cure_at(self, point):
        h1 = VMobject(color=BLACK)
        h1.set_points([[-1,-1,0], [-1,1,0], [1,1,0], [1,-1,0]])
        h1.set_points_as_corners([[-1,-1,0], [-1,1,0], [1,1,0], [1,-1,0]])
        h1.set_x(point[0])
        h1.set_y(point[1])
        return h1

    def construct(self):
        base_position = np.array([-5,-2,0])
        h0 = self.get_basic_cure_at(base_position)

        max_deg = 2
        for degree in range(1,max_deg):
            side_length = 2*(2*degree-1)
            move_to_next_deegre = np.array([side_length+1,0,0])
            move_up = np.array([0,2*side_length,0])
            move_right = np.array([2*side_length,0,0])
            move_down = np.array([0,-2*side_length,0])

            h1 = self.get_basic_cure_at(base_position)
            position_1st = base_position + move_to_next_deegre
            h2 = self.get_basic_cure_at(position_1st)
            position_2nd = position_1st + move_up
            h3 = self.get_basic_cure_at(position_2nd)
            position_3rd = position_2nd + move_right
            h4 = self.get_basic_cure_at(position_3rd)
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

            base_position = self.get_middle_point(position_1st, position_4th)


    def get_middle_point(self, p1, p2):
        return np.array([(p1[0]+p2[0])/2, (p1[1]+p2[1])/2])
