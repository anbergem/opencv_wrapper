import enum

import random


class ColorAttr(enum.EnumMeta):
    """
    See:
    https://stackoverflow.com/questions/47353555/how-to-get-random-value-of-attribute-of-enum-on-each-iteration/47353856
    """

    @property
    def RANDOM(self):
        return tuple(random.randint(0, 255) for _ in range(3))


class Color(enum.Enum, metaclass=ColorAttr):
    RED = (0, 0, 155)
    BLUE = (155, 0, 0)
    GREEN = (0, 155, 0)
    CYAN = (155, 155, 0)
    WHITE = (255, 255, 255)
