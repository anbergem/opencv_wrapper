import enum

import numpy as np


class ColorAttr(enum.EnumMeta):
    """
    See:
    https://stackoverflow.com/questions/47353555/how-to-get-random-value-of-attribute-of-enum-on-each-iteration/47353856
    """

    @property
    def RANDOM(self):
        return (*np.random.randint(0, 256, size=3),)


class Color(enum.Enum, metaclass=ColorAttr):
    RED = (0, 0, 155)
    BLUE = (155, 0, 0)
    GREEN = (0, 155, 0)
    CYAN = (155, 155, 0)
    WHITE = (255, 255, 255)
