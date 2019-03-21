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
    """
    Draw a circle on `img` at `center` with `radius`.

    :param img: The image to draw the circle
    :param center: The center at which to draw the circle
    :param radius: The radius of the circle
    :param color: The color of the circle.
    :param thickness: The thickness of the circle; can be -1 to fill the circle.
    """
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
    """
    Draw a line from `point1` to `point2` on `image`.

    :param image: The image to draw the line
    :param point1: The starting point
    :param point2: The ending point
    :param color: The color of the line
    :param thickness: The thickness of the line
    :param line_style: The line style to draw. For LineStyle.DASHED, only thickness
                       1 is currently supported.
    """
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
    """
    Draw a rectangle on `image`.

    :param image: The image to draw the rectangle
    :param rect: The rectangle to be drawn
    :param color: The color of the rectangle
    :param thickness: The thickness of the lines; can be -1 to fill the rectangle.
    :param line_style: The line style to draw. For LineStyle.DASHED, only thickness
                       1 is currently supported.
    """
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
    """
    Put `text` on `image` at `origin`.

    :param img: The image to draw the text
    :param text: The text to be drawn
    :param origin: The origin to start the text. The bottom of the first character
                   is set in the origin.
    :param color: The color of the text
    :param thickness: The thickness of the text
    :param scale: The scale of the text.
    """
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
    """
    Wait for a key event infinitely (if `delay` is 0) or `delay` amount of milliseconds.

    An alias for cv.waitKey(delay) & 0xFF. See cv.waitKey(delay) for further documentation.

    :param delay: Amount of milliseconds to wait, or 0 for infinitely.
    :return: The key pressed. Comparison of the key pressed can be found by `ord(str)`. For example
             `if wait_key(0) == ord('q'): continue`
    """
    return cv.waitKey(delay) & 0xFF
