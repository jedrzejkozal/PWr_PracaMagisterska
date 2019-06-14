from objects.Line import *
from objects.Circle import *
from objects.Text import *
from geometry.translation import *
from engine import *
from PIL import ImageFont


freesans_font = ImageFont.truetype('/usr/share/fonts/opentype/freefont/FreeSans.otf', size=18)
objects_list = [Text("h\u2080", 0, -1, font=freesans_font)]

sequence_len = 5
interval = 55
subindexes = ['\u2080', '\u2081', '\u2082', '\u2083', '\u2084', '\u2085', '\u2086']
for i in range(sequence_len+1):
    start_time = i*interval
    objects_list.append(Line((0,0), (10,0), start_time, -1, width=1, transformations=[(Line.extend, start_time, start_time+15, 3), (Line.show_arrow, start_time+14, start_time+15)]))
    objects_list.append(Line((0,20), (0,0), start_time, -1, width=1, transformations=[(Line.extend, start_time, start_time+15, 4.5), (Line.show_arrow, start_time+14, start_time+15)]))
    objects_list.append(Circle(25, start_time+35, -1, color='white'))
    objects_list.append(Text("x"+subindexes[i+1], 0, -1, font=freesans_font))
    objects_list.append(Text("h"+subindexes[i+1], start_time+35, -1, font=freesans_font))
    objects_list.append(Line((0,1), (0,0), start_time+40, -1, width=1, transformations=[(Line.extend, start_time+40, start_time+55, 5), (Line.show_arrow, start_time+54, start_time+55)]))
    objects_list.append(Text("y"+subindexes[i+1], start_time+56, -1, font=freesans_font))

line1_1_base = (35,200)
line1_2_base = (120,320)
circle1_base = (120,200)
x1_base = (113,330)
h1_base = (135,160)
line1_3_base = (120,170)
y1_base = (113,70)

trajectories = [constant_position((10,200-10), num_frames=1000),]

translation_base = (130,0)
translation_vec = (0,0)
for i in range(sequence_len+1):
    trajectories.append(constant_position(move_point_by_vector(line1_1_base, translation_vec), num_frames=1000))
    trajectories.append(constant_position(move_point_by_vector(line1_2_base, translation_vec), num_frames=1000))
    trajectories.append(constant_position(move_point_by_vector(circle1_base, translation_vec), num_frames=1000))
    trajectories.append(constant_position(move_point_by_vector(x1_base, translation_vec), num_frames=1000))
    trajectories.append(constant_position(move_point_by_vector(h1_base, translation_vec), num_frames=1000))
    trajectories.append(constant_position(move_point_by_vector(line1_3_base, translation_vec), num_frames=1000))
    trajectories.append(constant_position(move_point_by_vector(y1_base, translation_vec), num_frames=1000))
    translation_vec = move_point_by_vector(translation_vec, translation_base)

generate_gif(objects_list, trajectories, filename='samples/rnn.gif', size=(820, 400))
