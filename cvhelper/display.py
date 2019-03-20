from typing import Union, Tuple
from enum import Enum, auto

import cv2 as cv
import numpy as np

from .model import Point, Rect
from .utils import Color, _ensure_compatible_color
from .misc_functions import line_iterator


class LineStyle(Enum):
    SOLID = auto()
    DASHED = auto()


def circle(
    img: np.ndarray,
    center: Union[Point, Tuple[int, int]],
    radius: int,
    color: Union[int, Tuple[int, int, int], Color],
    thickness: int = 1,
):
    x, y = map(int, center)
    color = _ensure_compatible_color(color)
    cv.circle(img, (x, y), radius, color=color, thickness=thickness)


def line(
    image: np.ndarray,
    point1: Union[Point, Tuple[int, int]],
    point2: Union[Point, Tuple[int, int]],
    color: Union[int, Tuple[int, int, int], Color],
    thickness: int = 1,
    line_style: LineStyle = LineStyle.SOLID,
):
    if isinstance(point1, tuple) or isinstance(point1.x, float):
        point1 = Point(*map(int, point1))
    if isinstance(point2, tuple) or isinstance(point2.x, float):
        point2 = Point(*map(int, point2))

    color = _ensure_compatible_color(color)

    if line_style is LineStyle.SOLID:
        cv.line(image, (*point1,), (*point2,), color, thickness, cv.LINE_AA)
    elif line_style is LineStyle.DASHED:
        iterator = line_iterator(image, point1, point2)
        image[iterator[::2, 1], iterator[::2, 0]] = color
        image[iterator[1::4, 1], iterator[1::4, 0]] = color
    else:
        raise ValueError(f"unknown line style: {line_style}")


def rectangle(
    image: np.ndarray,
    rect: Rect,
    color: Union[Color, Tuple[int, int, int], int],
    thickness: int = 1,
    line_style: LineStyle = LineStyle.SOLID,
):
    color = _ensure_compatible_color(color)
    rect = Rect(*map(int, rect))
    if line_style is LineStyle.SOLID:
        cv.rectangle(image, *rect.aspoints, color, thickness)
    elif line_style is LineStyle.DASHED:
        line(image, rect.tl, rect.tr, color, line_style=line_style)
        line(image, rect.bl, rect.br, color, line_style=line_style)
        line(image, rect.tl, rect.bl, color, line_style=line_style)
        line(image, rect.tr, rect.br, color, line_style=line_style)
    else:
        raise ValueError(f"unknown line style: {line_style}")


def put_text(
    img: np.ndarray,
    text: str,
    origin: Union[Point, Tuple[int, int]],
    color: Union[int, Tuple[int, int, int], Color] = Color.RED,
    thickness: int = 1,
    scale: float = 1,
):
    color = _ensure_compatible_color(color)
    cv.putText(
        img,
        text,
        (*map(int, origin),),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness=thickness,
    )


def wait_key(delay: int) -> str:
    return cv.waitKey(delay) & 0xFF
