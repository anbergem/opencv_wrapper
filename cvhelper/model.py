from dataclasses import dataclass, astuple
from typing import Tuple, Union


@dataclass
class Point:
    x: float
    y: float

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
    x: float
    y: float
    width: float
    height: float

    def __init__(
        self, x: float, y: float, width: float, height: float, *, padding: float = 0
    ):
        self.x = x - padding
        self.y = y - padding
        self.width = width + padding * 2
        self.height = height + padding * 2

    def __iter__(self):
        return iter((self.x, self.y, self.width, self.height))

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Rect(
                self.x / other, self.y / other, self.width / other, self.height / other
            )
        return NotImplemented

    def __floordiv__(self, other):
        if isinstance(other, int):
            return Rect(
                self.x // other,
                self.y // other,
                self.width // other,
                self.height // other,
            )
        return NotImplemented

    def __contains__(self, point: Union[Point, Tuple[int, int]]):
        if isinstance(point, tuple):
            point = Point(*point)
        if isinstance(point, Point):
            return (
                self.x <= point.x < self.x + self.width
                and self.y <= point.y < self.y + self.height
            )
        raise ValueError("Must be called with a point or a 2-tuple (x, y)")

    @property
    def tl(self) -> Point:
        return Point(self.x, self.y)

    @property
    def tr(self) -> Point:
        return Point(self.x + self.width, self.y)

    @property
    def bl(self) -> Point:
        return Point(self.x, self.y + self.height)

    @property
    def br(self) -> Point:
        return Point(self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Point:
        return Point(self.x + (self.width / 2), self.y + (self.height / 2))

    @property
    def aspoints(self):
        return tuple(astuple(point) for point in (self.tl, self.br))

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def slice(self) -> Tuple[slice, slice]:
        return (
            slice(int(self.y), int(self.y) + int(self.height)),
            slice(int(self.x), int(self.x) + int(self.width)),
        )
