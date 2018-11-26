from dataclasses import dataclass, astuple
from typing import Union


@dataclass
class Point:
    x: Union[int, float]
    y: Union[int, float]

    def __add__(self, other):
        if self.__class__ is other.__class__:
            return Point(self.x + other.x, self.y + other.y)
        elif hasattr(other, "__len__") and len(other) == 2:
            return Point(self.x + other[0], self.y + other[1])
        return NotImplemented

    def __sub__(self, other):
        if self.__class__ is other.__class__:
            return Point(self.x - other.x, self.y - other.y)
        elif hasattr(other, "__len__") and len(other) == 2:
            return Point(self.x - other[0], self.y - other[1])
        return NotImplemented

    def __iter__(self):
        return iter((self.x, self.y))

    @classmethod
    def origo(cls):
        return cls(0, 0)


@dataclass()
class Rect:
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]

    def __init__(
        self,
        x: Union[int, float],
        y: Union[int, float],
        width: Union[int, float],
        height: Union[int, float],
        padding: Union[int, float] = 0,
    ):
        self.x = x - padding
        self.y = y - padding
        self.width = width + padding * 2
        self.height = height + padding * 2

    def __iter__(self):
        return iter((self.x, self.y, self.width, self.height))

    @property
    def tl(self):
        return Point(self.x, self.y)

    @property
    def br(self):
        return Point(self.x + self.width, self.y + self.height)

    @property
    def aspoints(self):
        return tuple(astuple(point) for point in (self.tl, self.br))

    @property
    def area(self):
        return self.width * self.height

    @property
    def slice(self):
        return slice(self.y, self.y + self.height), slice(self.x, self.x + self.width)
