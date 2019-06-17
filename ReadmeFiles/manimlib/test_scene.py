from manimlib.imports import *

class Test(Scene):

    def construct(self):
        y_coordinate = list(range(3, -3, -1))
        lines = [Line(np.array([0, y, 0]), np.array([1, y, 0]), color=BLACK) for y in y_coordinate]

        #self.play(ShowPartial(lines[0]))
        self.play(ShowCreation(lines[1]))
        self.play(Uncreate(lines[2]))
        self.play(DrawBorderThenFill(lines[3]))
        self.play(Write(lines[4]))
        self.play(ShowIncreasingSubsets(lines[5]))
