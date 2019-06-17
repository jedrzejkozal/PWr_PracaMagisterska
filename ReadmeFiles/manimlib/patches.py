from manimlib.imports import *
sys.path.insert(0,'/home/jkozal/Dokumenty/PWr/magisterka/magisterka/animations/manimlib/')
from renet_scene import *


img_width, img_height, channels = 6, 6, 3
patch_width, patch_height = 2, 2

class Patches(ReNetScene):

    def setup(self):
        self.camera.set_frame_center([0, 0, 0])
        self.camera.set_frame_width(self.camera.get_frame_width()+18)
        self.camera.set_frame_height(self.camera.get_frame_height()+9)


    def construct(self):
        img_position = np.array([-3,3,0])
        img = self.get_img(img_position, img_width, img_height, channels, patch_width)
        self.play_img_creation(img)

        patch_left_up = self.get_patch_pixels(img[0], 0, patch_width*patch_height*channels)
        patch_left = self.get_patch_pixels(img[0], 1, patch_width*patch_height*channels)
        patch_left_down = self.get_patch_pixels(img[0], 2, patch_width*patch_height*channels)
        left_up = self.simultaneous_movement_animations(patch_left_up, PhaseFlow, lambda x: np.array([-1,1,0]))
        left = self.simultaneous_movement_animations(patch_left, PhaseFlow, lambda x: np.array([-1,0,0]))
        left_down = self.simultaneous_movement_animations(patch_left_down, PhaseFlow, lambda x: np.array([-1,-1,0]))

        patch_up = self.get_patch_pixels(img[1], 0, patch_width*patch_height*channels)
        patch_middle = self.get_patch_pixels(img[1], 1, patch_width*patch_height*channels)
        patch_down = self.get_patch_pixels(img[1], 2, patch_width*patch_height*channels)
        up = self.simultaneous_movement_animations(patch_up, PhaseFlow, lambda x: np.array([0,1,0]))
        down = self.simultaneous_movement_animations(patch_down, PhaseFlow, lambda x: np.array([0,-1,0]))

        patch_right_up = self.get_patch_pixels(img[2], 0, patch_width*patch_height*channels)
        patch_right = self.get_patch_pixels(img[2], 1, patch_width*patch_height*channels)
        patch_right_down = self.get_patch_pixels(img[2], 2, patch_width*patch_height*channels)
        right_up = self.simultaneous_movement_animations(patch_right_up, PhaseFlow, lambda x: np.array([1,1,0]))
        right = self.simultaneous_movement_animations(patch_right, PhaseFlow, lambda x: np.array([1,0,0]))
        right_down = self.simultaneous_movement_animations(patch_right_down, PhaseFlow, lambda x: np.array([1,-1,0]))

        animations = left_up + left + left_down + up + down + right_up + right + right_down
        self.play(*animations)

        patches = [patch_left_up, patch_up, patch_right_up,
                    patch_left, patch_middle, patch_right,
                    patch_left_down, patch_down, patch_right_down]
        for patch in patches:
            self.play(*self.simultaneous_animations(patch, WiggleOutThenIn))



    def get_img(self, postion, img_width, img_height, channels, patch_width):
        columns = []
        for i in range(0, img_width, patch_width):
            columns.append(self.get_pixels_columns(img_height, patch_width, channels, postion+np.array([i,0,0])))
        return columns


    def get_pixels_columns(self, column_len, columns_width, channels, starting_positon, color_palete=[RED, GREEN, BLUE]):
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
