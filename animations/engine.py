from PIL import Image

from objects.Object import *


def generate_gif(objects_list, trajectories, filename='samples/sample.gif', size=(400,400)):
    __check_trajectories_contents_types(trajectories)
    __check_if_objects_list_len_and_trajectories_len_match(len(objects_list), len(trajectories))
    __check_if_trajectories_have_equal_len(trajectories)
    __check_if_all_objects_derive_from_Object_class(objects_list)

    number_of_frames = len(trajectories[0])
    img_width, img_height = size
    frames = []

    for frame_number in range(number_of_frames):
        img = Image.new('RGB', (img_width, img_height), (255, 255, 255))
        img = __add_all_objects_to_img(img, objects_list, trajectories, frame_number)
        __apply_transformations(objects_list, frame_number)
        frames.append(img)

    frames[0].save(filename, format='GIF', append_images=frames[1:], save_all=True, duration=2, loop=0)


def __check_trajectories_contents_types(trajectories):
    for t in trajectories:
        for point in t:
            assert type(point) is tuple , "trajectories must contain points saved as tuples, got: {}".format(type(point))
            assert len(point) == 2, "trajectories must contain 3 coordinates, got {}".format(len(point))


def __check_if_objects_list_len_and_trajectories_len_match(len_objects_list, len_trajectories):
    assert len_objects_list == len_trajectories, "trajectories len must be equal to objects_list len, objects_list len = {}, trajectories len = {}".format(len_objects_list, len_trajectories)


def __check_if_trajectories_have_equal_len(trajectories):
    for i in range(1, len(trajectories)):
        assert len(trajectories[i-1]) == len(trajectories[i]), "trajectories len must be equal"


def __check_if_all_objects_derive_from_Object_class(objects_list):
    for o in objects_list:
        assert issubclass(o.__class__, Object), "objects must derive form Object class"


def __add_all_objects_to_img(img, objects_list, trajectories, frame_number):
    for i in range(len(objects_list)):
        if objects_list[i].should_be_drawn(frame_number):
            img = objects_list[i].draw(img, trajectories[i][frame_number])
    return img


def __apply_transformations(objects_list, frame_number):
    for object in objects_list:
        object.apply_transformations(frame_number)
