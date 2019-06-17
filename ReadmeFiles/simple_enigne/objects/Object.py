from abc import ABC, abstractmethod
from math import inf

class Object(ABC):

    def __init__(self, appears_at, dissapears_at, color='black', transformations=[]):
        self.appears_at = appears_at

        if dissapears_at == -1: #-1 is the end of animation
            dissapears_at = 99999999
        self.dissapears_at = dissapears_at
        self.transformations = transformations
        self.color = color


    def should_be_drawn(self, frame_number):
        return self.appears_at <= frame_number and frame_number < self.dissapears_at


    def apply_transformations(self, frame_number):
        for transformation, starts_at, ends_at, *args in self.transformations:
            if ends_at == -1:
                ends_at = 99999999
            if starts_at <= frame_number and frame_number < ends_at:
                transformation.__call__(self, *args)


    @abstractmethod
    def draw(self, img, point):
        pass
