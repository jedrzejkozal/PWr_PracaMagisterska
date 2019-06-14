from PIL import ImageDraw

from objects.Object import *
from geometry.points_operations import *


class Text(Object):

    def __init__(self, string, *args, font=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.string = string
        self.font = font


    def draw(self, img, point):
        draw = ImageDraw.Draw(img)
        draw.text(point, text=self.string, fill=self.color, font=self.font)
        return img
