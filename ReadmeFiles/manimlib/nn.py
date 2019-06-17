from manimlib.imports import *


class NN(Scene):

    def setup(self):
        self.camera.set_frame_center([0, 0, 0])
        self.camera.set_frame_width(self.camera.get_frame_width()+2)
        self.camera.set_frame_height(self.camera.get_frame_height()+1.5)


    def construct(self):
        layer1 = self.get_layer(3, np.array([-5,2,0]))
        layer2 = self.get_layer(4, np.array([-2,3,0]))
        layer3 = self.get_layer(5, np.array([1,4,0]))
        layer4 = self.get_layer(4, np.array([4,3,0]))

        self.play(*self.simultaneous_animations(layer1, ShowCreation))
        self.play(*self.simultaneous_animations(layer2, ShowCreation))
        self.play(*self.simultaneous_animations(layer3, ShowCreation))
        self.play(*self.simultaneous_animations(layer4, ShowCreation))

        inputs = self.get_inputs(3, np.array([-5,2,0]))
        self.play(*self.simultaneous_animations(inputs, ShowCreation))
        activations1 = self.get_activations_for_layers(3, np.array([-5,2,0]), 4, np.array([-2,3,0]))
        self.play(*self.simultaneous_animations(activations1, ShowCreation))
        activations2 = self.get_activations_for_layers(4, np.array([-2,3,0]), 5, np.array([1,4,0]))
        self.play(*self.simultaneous_animations(activations2, ShowCreation))
        activations3 = self.get_activations_for_layers(5, np.array([1,4,0]), 4, np.array([4,3,0]))
        self.play(*self.simultaneous_animations(activations3, ShowCreation))
        outputs = self.get_outputs(4, np.array([4,3,0]))
        self.play(*self.simultaneous_animations(outputs, ShowCreation))


    def get_layer(self, num_neurons, start_position):
        neurons = []
        for i in range(num_neurons):
            neurons.append(self.get_circle_at(self.get_ith_neuron_pos(start_position, i)))
        return neurons


    def get_ith_neuron_pos(self, start, i):
        return start + np.array([0,-2*i,0])

    def simultaneous_animations(self, objects_list, animation):
        return tuple([animation(object) for object in objects_list])


    def get_circle_at(self, position, color=BLACK):
        c = Circle(color=BLACK)
        c.set_x(position[0])
        c.set_y(position[1])
        return c


    def get_activations_for_layers(self,
                num_neurons1, start_position1,
                num_neurons2, start_position2):
        activations = []
        for i in range(num_neurons1):
            for j in range(num_neurons2):
                start = self.get_ith_neuron_pos(start_position1, i) + np.array([0.5,0,0])
                end = self.get_ith_neuron_pos(start_position2, j) + np.array([-0.5,0,0])
                a = Line(start, end, color=BLACK)
                self.bring_to_back(a)
                activations.append(a)
        return activations


    def get_inputs(self, num_inputs, start_position):
        inputs = []
        for i in range(num_inputs):
            end = self.get_ith_neuron_pos(start_position, i) + np.array([-0.5,0,0])
            start = end + np.array([-1,0,0])
            inputs.append(Line(start, end, color=BLACK))
        return inputs


    def get_outputs(self, num_outputs, start_position):
        outputs = []
        for i in range(num_outputs):
            end = self.get_ith_neuron_pos(start_position, i) + np.array([1.5,0,0])
            start = end + np.array([-1,0,0])
            outputs.append(Line(start, end, color=BLACK))
        return outputs
