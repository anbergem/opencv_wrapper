import enum
import random
from typing import Union, Tuple

import numpy as np


class _ColorAttr(enum.EnumMeta):
    """
    See:
    https://stackoverflow.com/questions/47353555/how-to-get-random-value-of-attribute-of-enum-on-each-iteration/47353856
    """

    @property
    def RANDOM(self):
        return tuple(random.randint(0, 255) for _ in range(3))


class Color(enum.Enum, metaclass=_ColorAttr):
    """
    Color enum for predefined colors.

    Color.RANDOM returns a random color. Colors can be added together.
    """

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


CVColor = Union[Color, int, Tuple[int, int, int]]


def _ensure_color_int(color: CVColor) -> Tuple[int, int, int]:
    if isinstance(color, Color):
        color = color.value

    # cv drawing methods need python ints, not numpy ints
    bgr = np.ones(3, dtype=int) * color  # implicit broadcasting
    b, g, r = map(int, bgr)
    return b, g, r
