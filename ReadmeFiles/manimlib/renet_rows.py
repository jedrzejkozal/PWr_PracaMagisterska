from manimlib.imports import *
sys.path.insert(0,'/home/jkozal/Dokumenty/PWr/magisterka/magisterka/ReadmeFiles/manimlib/')
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
        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([3,0,0]))
        self.play(*move_last_right)

        move_last_down = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([0,-10,0]))
        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([4,0,0]))
        animations = move_last_down + move_middle_right
        self.play(*animations)

        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([11.5,0,0]))
        move_middle_down = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([0,-10,0]))
        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([5,0,0]))
        animations = move_last_right + move_middle_down + move_first_right
        self.play(*animations)

        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([7.5,0,0]))
        move_first_down = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([0,-10,0]))
        animations = move_middle_right + move_first_down
        self.play(*animations)

        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([3.5,0,0]))
        self.play(*move_first_right)

        c0 = self.get_circle_at(-0.5, 0, -1)
        c1 = self.get_circle_at(3, 0, -1)
        c2 = self.get_circle_at(6.5, 0, -1)

        self.computations_for_column((patch0, patch1, patch2), 0, circles=(c0,c1,c2))

        patch0 = self.get_patch_pixels(img[1], 0, patch_width*patch_height*channels)
        patch1 = self.get_patch_pixels(img[1], 1, patch_width*patch_height*channels)
        patch2 = self.get_patch_pixels(img[1], 2, patch_width*patch_height*channels)
        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([3,0,0]))
        self.play(*move_last_right)

        move_last_down = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([0,-9,0]))
        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([4,0,0]))
        animations = move_last_down + move_middle_right
        self.play(*animations)

        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([11.5,0,0]))
        move_middle_down = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([0,-9,0]))
        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([5,0,0]))
        animations = move_last_right + move_middle_down + move_first_right
        self.play(*animations)

        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([7.5,0,0]))
        move_first_down = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([0,-9,0]))
        animations = move_middle_right + move_first_down
        self.play(*animations)

        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([3.5,0,0]))
        self.play(*move_first_right)

        self.computations_for_column((patch0, patch1, patch2), 1)

        patch0 = self.get_patch_pixels(img[2], 0, patch_width*patch_height*channels)
        patch1 = self.get_patch_pixels(img[2], 1, patch_width*patch_height*channels)
        patch2 = self.get_patch_pixels(img[2], 2, patch_width*patch_height*channels)
        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([3,0,0]))
        self.play(*move_last_right)

        move_last_down = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([0,-8,0]))
        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([4,0,0]))
        animations = move_last_down + move_middle_right
        self.play(*animations)

        move_last_right = self.simultaneous_movement_animations(patch2, PhaseFlow, lambda x: np.array([11.5,0,0]))
        move_middle_down = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([0,-8,0]))
        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([5,0,0]))
        animations = move_last_right + move_middle_down + move_first_right
        self.play(*animations)

        move_middle_right = self.simultaneous_movement_animations(patch1, PhaseFlow, lambda x: np.array([7.5,0,0]))
        move_first_down = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([0,-8,0]))
        animations = move_middle_right + move_first_down
        self.play(*animations)

        move_first_right = self.simultaneous_movement_animations(patch0, PhaseFlow, lambda x: np.array([3.5,0,0]))
        self.play(*move_first_right)

        self.computations_for_column((patch0, patch1, patch2), 2)

        self.play(Uncreate(c0), Uncreate(c1), Uncreate(c2))



    def computations_for_column(self, patches, col, circles=None):
        patch0, patch1, patch2 = patches
        patch0_copy = self.get_pixels_rows(patch_width, patch_height, channels, np.array([-0.5,-7,0]))
        patch1_copy = self.get_pixels_rows(patch_width, patch_height, channels, np.array([3.5,-7,0]))
        patch2_copy = self.get_pixels_rows(patch_width, patch_height, channels, np.array([7.5,-7,0]))
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
        activation_patch0 = self.get_pixels_rows(1, 1, rnn_activations, np.array([-0.5,5,-0.5]), color_palete=[DARK_BLUE, DARK_BROWN])
        activation_patch1 = self.get_pixels_rows(1, 1, rnn_activations, np.array([3.5,5,-0.5]), color_palete=[DARK_BLUE, DARK_BROWN])
        activation_patch2 = self.get_pixels_rows(1, 1, rnn_activations, np.array([7,5,-0.5]), color_palete=[DARK_BLUE, DARK_BROWN])

        arrows = (arrow_in0, arrow_in1, arrow_in2, arrow_out0, arrow_out1, arrow_out2, arrow_rec1, arrow_rec2)
        patches = (patch0, patch1, patch2)
        activation_patches = (activation_patch0, activation_patch1, activation_patch2)
        self.show_rnn_sweep(arrows, patches, activation_patches)

        self.transport_activations(activation_patches, col, move_froward=True)

        move_all_patches_up = self.simultaneous_movement_animations(patch0_copy+patch1_copy+patch2_copy, PhaseFlow, lambda x: np.array([0,3,0]))
        self.play(*move_all_patches_up)

        arrow_rec1 = Arrow(np.array([2.5,0,-1]), np.array([0,0,-1]), color=BLACK)
        arrow_rec2 = Arrow(np.array([6,0,-1]), np.array([3.5,0,-1]), color=BLACK)

        activation_patch0 = self.get_pixels_rows(1, 1, rnn_activations, np.array([-0.5,5,-0.5]), color_palete=[TEAL_E, GREY])
        activation_patch1 = self.get_pixels_rows(1, 1, rnn_activations, np.array([3.5,5,-0.5]), color_palete=[TEAL_E, GREY])
        activation_patch2 = self.get_pixels_rows(1, 1, rnn_activations, np.array([7.5,5,-0.5]), color_palete=[TEAL_E, GREY])

        arrows = (arrow_in2, arrow_in1, arrow_in0, arrow_out2, arrow_out1, arrow_out0, arrow_rec2, arrow_rec1)
        patches = (patch2_copy, patch1_copy, patch0_copy)
        activation_patches = (activation_patch2, activation_patch1, activation_patch0)
        self.show_rnn_sweep(arrows, patches, activation_patches)

        self.transport_activations(tuple(reversed(activation_patches)), col, move_froward=False)


    def transport_activations(self, activation_patches, row, move_froward=True):
        activation_patch0, activation_patch1, activation_patch2 = activation_patches
        if move_froward:
            z = 1
        else:
            z = -1

        animations = self.simultaneous_movement_animations(activation_patch2, PhaseFlow, lambda x: np.array([5.5,0,z]))
        self.play(*animations)
        animations = self.simultaneous_movement_animations(activation_patch1, PhaseFlow, lambda x: np.array([8,0,z])) + self.simultaneous_movement_animations(activation_patch2, PhaseFlow, lambda x: np.array([0,-4-row,0]))
        self.play(*animations)
        animations = self.simultaneous_movement_animations(activation_patch0, PhaseFlow, lambda x: np.array([11,0,z])) + self.simultaneous_movement_animations(activation_patch1, PhaseFlow, lambda x: np.array([0,-4-row,0]))
        self.play(*animations)
        animations = self.simultaneous_movement_animations(activation_patch0, PhaseFlow, lambda x: np.array([0,-4-row,0]))
        self.play(*animations)


    def get_img(self, postion, img_width, img_height, channels, patch_width):
        rows = []
        for i in range(0, img_width, patch_width):
            rows.append(self.get_pixels_rows(img_height, patch_height, channels, postion+np.array([0,-1*i,0])))
        return rows


    def get_pixels_rows(self, row_len, rows_height, channels, starting_positon, color_palete=[YELLOW, PURPLE, ORANGE, PINK]):
        pixels = []
        for i in range(row_len):
                for channel in range(channels):
                    p = self.get_pixel_at(starting_positon[0]+i,
                                            starting_positon[1],
                                            starting_positon[2]-channel,
                                            color=color_palete[channel])
                    pixels.append(p)
        return pixels
