from PIL import ImageDraw

from objects.Object import *


class Circle(Object):

    def __init__(self, radius, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.radius = radius


    def draw(self, img, point):
        draw = ImageDraw.Draw(img)
        draw.ellipse((point[0] - self.radius, point[1] - self.radius,
                        point[0] + self.radius, point[1] + self.radius),
                        fill=self.color,
                        outline='black')
        return img
