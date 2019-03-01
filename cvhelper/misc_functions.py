from typing import Union

import cv2 as cv
import numpy as np

from .model import Point


def norm(input: Union[Point, np.ndarray]):
    """Returns the L2 norm"""
    if isinstance(input, Point):
        return cv.norm((*input,))
    else:
        return cv.norm(input)
