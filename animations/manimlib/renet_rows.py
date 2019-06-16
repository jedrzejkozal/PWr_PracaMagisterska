from manimlib.imports import *
sys.path.insert(0,'/home/jkozal/Dokumenty/PWr/magisterka/magisterka/animations/manimlib/')
from renet_scene import *


img_width, img_height, channels = 3, 3, 4
patch_width, patch_height = 1, 1

class ReNetRows(ReNetScene):

    def setup(self):
        self.camera.set_frame_center([1, -1, 0])
        self.camera.set_frame_width(self.camera.get_frame_width()+18)
        self.camera.set_frame_height(self.camera.get_frame_height()+9)


    def construct(self):
        img_position = np.array([-9,3,0])
        img = self.get_img(img_position, img_width, img_height, channels, patch_width)
        self.play_img_creation(img)

        patch0 = self.get_patch_pixels(img[0], 0, patch_width*patch_height*channels)
        patch1 = self.get_patch_pixels(img[0], 1, patch_width*patch_height*channels)
        patch2 = self.get_patch_pixels(img[0], 2, patch_width*patch_height*channels)
        move_last_down = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([0,-6,0]))
        self.play(*move_last_down)
