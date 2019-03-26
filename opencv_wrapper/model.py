from dataclasses import dataclass
from typing import Tuple, Union, Iterator, cast
import builtins

import numpy as np
import cv2 as cv


@dataclass
class Point:
    """
    Model class for a point. Point can be added, subtracted and iterated over, yielding
    x, y.
    """

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

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y))

    @classmethod
    def origin(cls) -> "Point":
        """
        :return: Return the origin point, Point(0, 0)
        """
        return cls(0, 0)

    @property
    def norm(self) -> float:
        """
        Return the absolute L2 norm of the point. Alias for `cvw.norm(point)`.

        :return: The absolute L2 norm of the point
        """
        from .misc_functions import norm as norm_func

        return norm_func(self)


CVPoint = Union[Point, Tuple[int, int]]


@dataclass
class Rect:
    """
    Model class of a rectangle.
    """

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

    def __contains__(self, point: CVPoint):
        if isinstance(point, tuple):
            point = Point(*point)
        if isinstance(point, Point):
            return (
                self.x <= point.x <= self.x + self.width
                and self.y <= point.y <= self.y + self.height
            )
        raise ValueError("Must be called with a point or a 2-tuple (x, y)")

    @property
    def tl(self) -> Point:
        """
        :return: The top-left corner of the rectangle.
        """
        return Point(self.x, self.y)

    @property
    def tr(self) -> Point:
        """
        :return: The top-right corner of the rectangle.
        """
        return Point(self.x + self.width, self.y)

    @property
    def bl(self) -> Point:
        """
        :return: The bottom-left corner of the rectangle.
        """
        return Point(self.x, self.y + self.height)

    @property
    def br(self) -> Point:
        """
        :return: The bottom-right corner of the rectangle.
        """
        return Point(self.x + self.width, self.y + self.height)

    @property
    def center(self) -> Point:
        """
        :return: The center point of the rectangle.
        """
        return Point(self.x + (self.width / 2), self.y + (self.height / 2))

    @property
    def aspoints(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Yields the rectangle as top-left and bottom-right points, as used in cv2.rectangle.

        :return: The top-left and bottom-right corners of the rectangle as two-tuples.
        """
        tl = cast(Tuple[float, float], tuple(self.tl))
        br = cast(Tuple[float, float], tuple(self.br))
        return tl, br

    @property
    def area(self) -> float:
        """
        :return: The area of the rectangle
        """
        return self.width * self.height

    @property
    def slice(self) -> Tuple[builtins.slice, builtins.slice]:
        """
        Creates a slice of the rectangle, to be used on a 2-D numpy array-

        For example `image[rect.slice] = 255` will fill the area represented by
        the rectangle as white, in a gray-scale image.

        :return: The slice of the rectangle.
        """
        return (
            slice(int(self.y), int(self.y) + int(self.height)),
            slice(int(self.x), int(self.x) + int(self.width)),
        )


CVRect = Union[Rect, Tuple[int, int, int, int]]


class Contour:
    def __init__(self, points):
        """
        :param points: points from cv.findContours()
        """
        self._points = points
        self._moments = None
        self._bounding_rect = None

    @property
    def points(self) -> np.ndarray:
        """
        Return the contour points as would be returned from cv.findContours().

        :return: The contour points.
        """
        return self._points

    @property
    def area(self) -> float:
        """
        Return the area computed from cv.moments(points).

        :return: The area of the contour
        """
        if self._moments is None:
            self._moments = cv.moments(self.points)
        return self._moments["m00"]

    @property
    def bounding_rect(self) -> Rect:
        """
        Return the bounding rectangle around the contour. Uses cv.boundingRect(points).

        :return: The bounding rectangle of the contour
        """
        if self._bounding_rect is None:
            self._bounding_rect = Rect(*cv.boundingRect(self.points))
        return self._bounding_rect

    @property
    def center(self) -> Point:
        """
        Return the center point of the area. Due to skewed densities, the center
        of the bounding rectangle is preferred to the center from moments.

        :return: The center of the bounding rectangle
        """
        return self.bounding_rect.center

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.points[key, 0]
        if len(key) > 2:
            raise ValueError(f"Too many indices: {len(key)}")
        return self.points[key[0], 0, key[1]]

    def __setitem__(self, key, value):
        if isinstance(key, int):
            self.points[key, 0] = value
        if len(key) > 2:
            raise ValueError(f"Too many indices: {len(key)}")
        self.points[key[0], 0, key[1]] = value
