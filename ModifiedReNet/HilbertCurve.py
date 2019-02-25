import numpy as np

from CommandsGenerator import *

class HilbertCurve(object):

    def __init__(self):
        self.left_rotation_matrix = np.array([[0, -1], [1, 0]])
        self.right_rotation_matrix = np.array([[0, 1], [-1, 0]])


    def convert_2D_to_1D(self, img):
        vec = self.__traverse_img(img)
        return np.array(vec)


    def get_indexes_vec(self, side_length):
        dummy_img = self.__get_dummy_img(side_length)
        return self.__traverse_img(dummy_img)


    def __get_dummy_img(self, side_length):
        result = []

        for i in range(side_length):
            for j in range(side_length):
                result.append([i, j])

        result_matrix = np.array(result)
        return result_matrix.reshape((side_length, side_length, 2))


    def __traverse_img(self, img):
        self.__check_shape(img)

        self.direction = np.reshape(np.array([[1], [0]]), (2,1))
        self.curr_position = np.array([[0], [img.shape[0]-1]])

        curve_deegre = np.log2(img.shape[0])
        commands = self.__generate_commands(curve_deegre)

        return self.__execute_all_commands(commands, img)


    def __check_shape(self, img):
        if img.shape[0] != img.shape[1] or not self.__is_power_of_2(img.shape[0]):
            raise ValueError("Invalid img shape: {}".format(img.shape))


    def __is_power_of_2(self, number):
        return np.equal(np.mod(np.log2(number), 1), 0)


    def __generate_commands(self, deegre):
        g = CommandsGenerator()
        return g.generate_commands(deegre)


    def __execute_all_commands(self, commands, img):
        result = []
        result.append(img[self.curr_position[1], self.curr_position[0]])

        for command in commands:
            self.__execute_single_command(command, img, result)
            
        return result


    def __execute_single_command(self, command, img, result):
        if command is "left":
            self.__rotate_left()
        elif command is "right":
            self.__rotate_right()
        else: # "F"
            self.__move_forward()
            result.append(img[self.curr_position[1], self.curr_position[0]])


    def __rotate_left(self):
        self.direction = np.matmul(self.left_rotation_matrix, self.direction)


    def __rotate_right(self):
        self.direction = np.matmul(self.right_rotation_matrix, self.direction)


    def __move_forward(self):
        self.curr_position[0] += self.direction[0]
        self.curr_position[1] -= self.direction[1]
