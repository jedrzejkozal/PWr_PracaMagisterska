from objects.Eclipse import *
from geometry.translation import *
from engine import *


objects_list = [Eclipse(40, 0, -1)]
eclipse_movement = move_object_from_to((0,0),(400,400), num_frames=100)
trajectories = [eclipse_movement + list(reversed(eclipse_movement))]

generate_gif(objects_list, trajectories, filename='samples/moving_ball.gif')
