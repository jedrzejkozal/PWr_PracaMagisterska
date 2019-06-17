from objects.Object import *


class ObjectMock(Object):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.drawn = 0
        self.transformations_applied = 0


    def draw(self, img, point):
        assert type(point) == tuple
        assert len(point) == 2
        self.drawn += 1
        return img


    def dummy_transformation(self):
        self.transformations_applied += 1
