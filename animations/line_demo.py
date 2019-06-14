from objects.Line import *
from geometry.translation import *
from engine import *

transformations = [(Line.rotation, 0, -1, 0.01), (Line.extend, 0, -1, 1)]
objects_list = [Line((0,0), (20,0), 0, -1, transformations)]
line_movement = move_object_from_to((200,200),(200,200), num_frames=1000)
trajectories = [line_movement + list(reversed(line_movement))]

generate_gif(objects_list, trajectories, filename='samples/moving_line.gif')
