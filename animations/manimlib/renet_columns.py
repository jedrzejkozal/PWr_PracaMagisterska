from manimlib.imports import *
sys.path.insert(0,'/home/jkozal/Dokumenty/PWr/magisterka/magisterka/animations/manimlib/')
from renet_scene import *


img_width, img_height, channels = 6, 6, 3
patch_width, patch_height = 2, 2


class ReNetColumns(ReNetScene):

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

        move_middle_down = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([0,-8,0]))
        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([16,0,0]))
        animations = move_middle_down + move_last_right
        self.play(*animations)

        move_first_down = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([0,-10,0]))
        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([12,0,0]))
        animations = move_first_down + move_middle_right
        self.play(*animations)

        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([8,0,0]))
        self.play(*move_first_right)

        c0 = self.get_circle_at(-0.5, 0, -1)
        c1 = self.get_circle_at(3, 0, -1)
        c2 = self.get_circle_at(6.5, 0, -1)

        self.computations_for_column((patch0, patch1, patch2), 0, circles=(c0,c1,c2))

        patch0 = self.get_patch_pixels(img[1], 0, patch_width*patch_height*channels)
        patch1 = self.get_patch_pixels(img[1], 1, patch_width*patch_height*channels)
        patch2 = self.get_patch_pixels(img[1], 2, patch_width*patch_height*channels)
        move_last_down = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([0,-6,0]))
        self.play(*move_last_down)

        move_middle_down = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([0,-8,0]))
        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([14,0,0]))
        animations = move_middle_down + move_last_right
        self.play(*animations)

        move_first_down = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([0,-10,0]))
        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([10,0,0]))
        animations = move_first_down + move_middle_right
        self.play(*animations)

        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([6,0,0]))
        self.play(*move_first_right)

        self.computations_for_column((patch0, patch1, patch2), 1)

        patch0 = self.get_patch_pixels(img[2], 0, patch_width*patch_height*channels)
        patch1 = self.get_patch_pixels(img[2], 1, patch_width*patch_height*channels)
        patch2 = self.get_patch_pixels(img[2], 2, patch_width*patch_height*channels)
        move_last_down = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([0,-6,0]))
        self.play(*move_last_down)

        move_middle_down = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([0,-8,0]))
        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([12,0,0]))
        animations = move_middle_down + move_last_right
        self.play(*animations)

        move_first_down = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([0,-10,0]))
        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([8,0,0]))
        animations = move_first_down + move_middle_right
        self.play(*animations)

        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([4,0,0]))
        self.play(*move_first_right)

        self.computations_for_column((patch0, patch1, patch2), 2)

        self.play(Uncreate(c0), Uncreate(c1), Uncreate(c2))


    def computations_for_column(self, patches, col, circles=None):
        patch0, patch1, patch2 = patches
        patch0_copy = self.get_pixels_columns(patch_width, patch_height, channels, np.array([-1,-7,0]))
        patch1_copy = self.get_pixels_columns(patch_width, patch_height, channels, np.array([3,-7,0]))
        patch2_copy = self.get_pixels_columns(patch_width, patch_height, channels, np.array([7,-7,0]))
        move_all_patches_up = self.simultaneous_movement_animations(patch0+patch1+patch2, PhaseFlow, lambda x: np.array([0,3,0]))
        patches_copy = self.simultaneous_animations(patch0_copy+patch1_copy+patch2_copy, Write)
        animations = move_all_patches_up + patches_copy
        self.play(*animations)

        if circles is not None:
            c0, c1, c2 = circles
            self.play(Write(c0), Write(c1), Write(c2))

        arrow_in0 = Arrow(np.array([-0.5,-3,-1]), np.array([-0.5,-1,-1]), color=BLACK)
        arrow_in1 = Arrow(np.array([3.25,-3,-1]), np.array([3.25,-1,-1]), color=BLACK)
        arrow_in2 = Arrow(np.array([6.75,-3,-1]), np.array([6.75,-1,-1]), color=BLACK)
        arrow_out0 = Arrow(np.array([-0.5,1,-1]), np.array([-0.5,3,-1]), color=BLACK)
        arrow_out1 = Arrow(np.array([3.25,1,-1]), np.array([3.25,3,-1]), color=BLACK)
        arrow_out2 = Arrow(np.array([6.75,1,-1]), np.array([6.75,3,-1]), color=BLACK)
        arrow_rec1 = Arrow(np.array([0,0,-1]), np.array([2.5,0,-1]), color=BLACK)
        arrow_rec2 = Arrow(np.array([3.5,0,-1]), np.array([6,0,-1]), color=BLACK)

        rnn_activations = 2
        activation_patch0 = self.get_pixels_columns(1, 1, rnn_activations, np.array([-0.5,5,-0.5]), color_palete=[YELLOW, PURPLE])
        activation_patch1 = self.get_pixels_columns(1, 1, rnn_activations, np.array([3.5,5,-0.5]), color_palete=[YELLOW, PURPLE])
        activation_patch2 = self.get_pixels_columns(1, 1, rnn_activations, np.array([7,5,-0.5]), color_palete=[YELLOW, PURPLE])

        arrows = (arrow_in0, arrow_in1, arrow_in2, arrow_out0, arrow_out1, arrow_out2, arrow_rec1, arrow_rec2)
        patches = (patch0, patch1, patch2)
        activation_patches = (activation_patch0, activation_patch1, activation_patch2)
        self.show_rnn_sweep(arrows, patches, activation_patches)

        self.transport_activations(activation_patches, col, move_froward=True)

        move_all_patches_up = self.simultaneous_movement_animations(patch0_copy+patch1_copy+patch2_copy, PhaseFlow, lambda x: np.array([0,3,0]))
        self.play(*move_all_patches_up)

        arrow_rec1 = Arrow(np.array([2.5,0,-1]), np.array([0,0,-1]), color=BLACK)
        arrow_rec2 = Arrow(np.array([6,0,-1]), np.array([3.5,0,-1]), color=BLACK)

        activation_patch0 = self.get_pixels_columns(1, 1, rnn_activations, np.array([-0.5,5,-0.5]), color_palete=[ORANGE, PINK])
        activation_patch1 = self.get_pixels_columns(1, 1, rnn_activations, np.array([3.5,5,-0.5]), color_palete=[ORANGE, PINK])
        activation_patch2 = self.get_pixels_columns(1, 1, rnn_activations, np.array([7,5,-0.5]), color_palete=[ORANGE, PINK])

        arrows = (arrow_in2, arrow_in1, arrow_in0, arrow_out2, arrow_out1, arrow_out0, arrow_rec2, arrow_rec1)
        patches = (patch2_copy, patch1_copy, patch0_copy)
        activation_patches = (activation_patch2, activation_patch1, activation_patch0)
        self.show_rnn_sweep(arrows, patches, activation_patches)

        self.transport_activations(tuple(reversed(activation_patches)), col, move_froward=False)


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


    def transport_activations(self, activation_patches, col, move_froward=True):
        activation_patch0, activation_patch1, activation_patch2 = activation_patches
        if move_froward:
            z = 1
        else:
            z = -1

        animations = self.simultaneous_movement_animations(activation_patch2, PhaseFlow, lambda x: np.array([3.5+col,0,z]))
        self.play(*animations)
        animations = self.simultaneous_movement_animations(activation_patch1, PhaseFlow, lambda x: np.array([7+col,0,z])) + self.simultaneous_movement_animations(activation_patch2, PhaseFlow, lambda x: np.array([0,-6,0]))
        self.play(*animations)
        animations = self.simultaneous_movement_animations(activation_patch0, PhaseFlow, lambda x: np.array([11+col,0,z])) + self.simultaneous_movement_animations(activation_patch1, PhaseFlow, lambda x: np.array([0,-5,0]))
        self.play(*animations)
        animations = self.simultaneous_movement_animations(activation_patch0, PhaseFlow, lambda x: np.array([0,-4,0]))
        self.play(*animations)


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
