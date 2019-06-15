from manimlib.imports import *

class Rnn(Scene):

    def setup(self):
        self.camera.set_frame_width(self.camera.get_frame_width()+3)

    def construct(self):
        h0 = TextMobject("$h_0$", color=BLACK)
        h0.set_x(-7.5)
        self.add(h0)

        translation = 3.5
        sequence_len = 4
        for i in range(sequence_len):
            x = TextMobject("$x_"+str(i+1)+"$", color=BLACK)
            x.set_x(-4 + i*translation)
            x.set_y(-3.2)
            self.add(x)

        for i in range(sequence_len):
            translation_x = i*translation
            line1 = Arrow(np.array([-7+translation_x,0,0]), np.array([-4.5+translation_x,0,0]), color=BLACK)
            line2 = Arrow(np.array([-4+translation_x,-3,0]), np.array([-4+translation_x,-0.5,0]), color=BLACK)

            circle = Circle(color=BLACK)
            circle.set_x(-4+translation_x)
            circle.set_y(0)
            h1 = TextMobject("$h_"+str(i+1)+"$", color=BLACK)
            h1.set_x(-3.3+translation_x)
            h1.set_y(0.5)
            line3 = Arrow(np.array([-4.0+translation_x, 0.5, 0]), np.array([-4.0+translation_x, 3, 0]), color=BLACK)
            y1 = TextMobject("$y_"+str(i+1)+"$", color=BLACK)
            y1.set_x(-4+translation_x)
            y1.set_y(3.2)

            self.play(ShowCreation(line1), ShowCreation(line2))
            self.play(FadeIn(circle), FadeIn(h1))
            self.play(ShowCreation(line3), FadeIn(y1))
