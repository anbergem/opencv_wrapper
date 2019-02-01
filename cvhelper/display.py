from typing import Union, Tuple

import cv2 as cv
import numpy as np

from .model import Point, Rect
from .utils import Color


def circle(
    img: np.ndarray,
    center: Union[Point, Tuple[int, int]],
    radius: int,
    color: Union[int, Tuple[int, int, int], Color],
    thickness: int = 1,
):
    x, y = map(int, center)
    if isinstance(color, Color):
        color = color.value
    cv.circle(img, (x, y), radius, color=color, thickness=thickness)


def line(
    image: np.ndarray,
    point1: Union[Point, Tuple[int, int]],
    point2: Union[Point, Tuple[int, int]],
    color: Union[int, Tuple[int, int, int], Color],
    thickness: int = 1,
):
    if isinstance(color, Color):
        color = color.value
    cv.line(image, (*point1,), (*point2,), color, thickness, cv.LINE_AA)


def rectangle(
    image: np.ndarray,
    rectangle: Rect,
    color: Union[Color, Tuple[int, ...], int],
    thickness: int = 1,
):
    if isinstance(color, Color):
        color = color.value
    cv.rectangle(image, *rectangle.aspoints, color, thickness)


def put_text(
    img: np.ndarray,
    text: str,
    origin: Union[Point, Tuple[int, int]],
    color: Union[int, Tuple[int, int, int], Color] = Color.RED,
    thickness: Union[int, float] = 1,
    scale: Union[int, float] = 1,
):
    if isinstance(color, Color):
        color = color.value
    cv.putText(
        img,
        text,
        (*origin,),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness=thickness,
    )


def wait_key(delay: int, key="q") -> bool:
    return cv.waitKey(delay) & 0xFF == ord(key)
