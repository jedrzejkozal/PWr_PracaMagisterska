from manimlib.imports import *


class CubeConfig(Cube):
    CONFIG = {
        "fill_opacity": 0.75,
        "fill_color": RED,
        "stroke_width": 0,
        "side_length": 1,
    }

    def __init__(self, color=RED):
        super().__init__()
        self.fill_color = color
        self.init_colors()


class ReNetScene(ThreeDScene):

    def get_pixel_at(self, x, y, z, color=RED):
        c = CubeConfig(color=color)
        c.set_x(x)
        c.set_y(y)
        c.set_z(z)
        return c


    def play_img_creation(self, img):
        animations = tuple()
        for column in img:
            animations = animations + self.simultaneous_animations(column, ShowCreation)
        self.play(*animations)


    def simultaneous_animations(self, objects_list, animation):
        return tuple([animation(object) for object in objects_list])


    #most movement animations have some argument before object
    def simultaneous_movement_animations(self, objects_list, animation, arg):
        return tuple([animation(arg, object) for object in objects_list])


    def get_patch_pixels(self, column, i, patch_size):
        return column[i*patch_size:i*patch_size+patch_size]


    def get_circle_at(self, x, y, z, color=BLACK):
        c = Circle(color=BLACK)
        c.set_x(x)
        c.set_y(y)
        return c


    def show_rnn_sweep(self, arrows, patches, activation_patches):
        arrow_in0, arrow_in1, arrow_in2, arrow_out0, arrow_out1, arrow_out2, arrow_rec1, arrow_rec2 = arrows
        patch0, patch1, patch2 = patches
        activation_patch0, activation_patch1, activation_patch2 = activation_patches

        animations = (ShowCreation(arrow_in0),) + self.simultaneous_animations(patch0, VFadeOut)
        self.play(*animations)
        animations = (ShowCreation(arrow_in1), ShowCreation(arrow_out0), ShowCreation(arrow_rec1),
                        FadeOut(arrow_in0)) + self.simultaneous_animations(patch1, VFadeOut) + self.simultaneous_animations(activation_patch0, VFadeIn)
        self.play(*animations)
        animations = (FadeOut(arrow_in1), FadeOut(arrow_out0), FadeOut(arrow_rec1),
                        ShowCreation(arrow_in2), ShowCreation(arrow_out1),
                        ShowCreation(arrow_rec2)) + self.simultaneous_animations(patch2, VFadeOut) + self.simultaneous_animations(activation_patch1, VFadeIn)
        self.play(*animations)
        animations = (FadeOut(arrow_in2), FadeOut(arrow_out1), FadeOut(arrow_rec2), ShowCreation(arrow_out2)) + self.simultaneous_animations(activation_patch2, VFadeIn)
        self.play(*animations)
        animations = (FadeOut(arrow_out2),)
        self.play(*animations)
