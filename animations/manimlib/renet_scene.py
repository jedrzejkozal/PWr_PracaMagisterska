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

    def get_img(self, postion, img_width, img_height, channels, patch_width):
        columns = []
        for i in range(0, img_width, patch_width):
            columns.append(self.get_pixels_columns(img_height, patch_width, channels, postion+np.array([i,0,0])))
        return columns


    def get_pixels_columns(self, column_len, columns_width, channels, starting_positon, color_palete=[YELLOW, PURPLE, ORANGE, PINK]):
        pixels = []
        for i in range(column_len):
            for w in range(columns_width):
                for channel in range(channels):
                    p = self.get_pixel_at(starting_positon[0]+w,
                                            starting_positon[1]-i,
                                            starting_positon[2]-channel,
                                            color=color_palete[channel])
                    pixels.append(p)
        return pixels


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
