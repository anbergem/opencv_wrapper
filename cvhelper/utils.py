import enum
import random
import operator
from typing import Union, Tuple

import numpy as np

from .image_operations import normalize


class _ColorAttr(enum.EnumMeta):
    """
    See:
    https://stackoverflow.com/questions/47353555/how-to-get-random-value-of-attribute-of-enum-on-each-iteration/47353856
    """

    @property
    def RANDOM(self):
        return tuple(random.randint(0, 255) for _ in range(3))


class Color(enum.Enum, metaclass=_ColorAttr):
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    BLUE = (200, 0, 0)
    GREEN = (0, 200, 0)
    RED = (0, 0, 200)
    CYAN = (255, 255, 0)
    MAGENTA = (255, 0, 255)
    YELLOW = (0, 255, 255)

    def __add__(self, other):
        if isinstance(other, Color):
            sum = np.asarray(self.value) + np.asarray(other.value)
            normalized = 255 * sum / (sum.max())
            return tuple(map(int, normalized))
        return NotImplemented


def _ensure_compatible_color(color: Union[int, Tuple[int, int, int], Color]):
    if isinstance(color, Color):
        color = color.value

    # cv drawing methods need python ints, not numpy ints
    if hasattr(color, "__len__"):
        if not isinstance(color[0], int):
            color = tuple(map(int, color))
    else:
        color = int(color)

    return color
