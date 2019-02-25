import numpy as np

from GenerateCommands import *

class HilbertCurve(object):

    def __init__(self):
        self.left_rotation_matrix = np.array([[0, -1], [1, 0]])
        self.right_rotation_matrix = np.array([[0, 1], [-1, 0]])

    def hilbert_curve(self, img):
        self.check_shape(img)

        side_length = img.shape[0]
        curve_deegre = np.log2(side_length)

        self.direction = np.reshape(np.array([[1], [0]]), (2,1))
        self.curr_position = np.array([[0], [img.shape[0]-1]])

        result = []
        result.append(img[self.curr_position[1], self.curr_position[0]])

        commands = self.generate_commands(curve_deegre)

        for command in commands:
            if command is "left":
                self.rotate_left()
            elif command is "right":
                self.rotate_right()
            else: #"F"
                self.move_forward()
                result.append(img[self.curr_position[1], self.curr_position[0]])

        return np.array(result)


    def check_shape(self, img):
        if img.shape[0] != img.shape[1] or not self.is_power_of_2(img.shape[0]):
            raise ValueError("Invalid img shape: {}".format(img.shape))


    def is_power_of_2(self, number):
        return np.equal(np.mod(np.log2(number), 1), 0)


    def generate_commands(self, deegre):
        g = GenerateCommands()
        return g.generate_commands(deegre)


    def rotate_left(self):
        self.direction = np.matmul(self.left_rotation_matrix, self.direction)


    def rotate_right(self):
        self.direction = np.matmul(self.right_rotation_matrix, self.direction)


    def move_forward(self):
        self.curr_position[0] += self.direction[0]
        self.curr_position[1] -= self.direction[1]
