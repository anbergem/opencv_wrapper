from typing import Union, Tuple

import cv2 as cv
import numpy as np

from .model import Point, Rect
from .utils import Color


def rectangle(
    image: np.ndarray,
    rectangle: Rect,
    color: Union[Color, Tuple[int, ...]],
    thickness: int = 1,
):
    if isinstance(color, Color):
        color = color.value
    cv.rectangle(image, *rectangle.aspoints, color, thickness)


def put_text(
    img: np.ndarray,
    text: str,
    origin: Point,
    color: Color = Color.RED,
    scale: Union[int, float] = 1,
) -> None:
    cv.putText(img, text, (*origin,), cv.FONT_HERSHEY_SIMPLEX, scale, color.value)


def wait_key(delay: int, key="q") -> bool:
    return cv.waitKey(delay) & 0xFF == ord(key)
