import warnings
from enum import Enum, auto
from typing import Union, Tuple, Sequence

import cv2 as cv
import numpy as np

from .misc_functions import line_iterator
from .model import Point, CVPoint, Rect, Contour, CVRect
from .utils import CVColor, _ensure_color_int


class LineStyle(Enum):
    SOLID = auto()
    DASHED = auto()


def circle(
    image: np.ndarray,
    center: Union[Point, Tuple[int, int]],
    radius: int,
    color: CVColor,
    thickness: int = 1,
):
    """
    Draw a circle on `image` at `center` with `radius`.

    :param image: The image to draw the circle
    :param center: The center at which to draw the circle
    :param radius: The radius of the circle
    :param color: The color of the circle.
    :param thickness: The thickness of the circle; can be -1 to fill the circle.
    """
    _warn_point_outside_image(image, center)
    x, y = map(int, center)
    color = _ensure_color_int(color)
    cv.circle(image, (x, y), radius, color=color, thickness=thickness)


def line(
    image: np.ndarray,
    point1: Union[Point, Tuple[int, int]],
    point2: Union[Point, Tuple[int, int]],
    color: CVColor,
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
    _warn_point_outside_image(image, point1)
    _warn_point_outside_image(image, point2)
    point1 = Point(*map(int, point1))
    point2 = Point(*map(int, point2))

    color = _ensure_color_int(color)

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
    color: CVColor,
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
    _warn_rect_outside_image(image, rect)
    color = _ensure_color_int(color)
    rect = Rect(*map(int, rect))
    if line_style is LineStyle.SOLID:
        cv.rectangle(image, *rect.cartesian_corners(), color, thickness)
    elif line_style is LineStyle.DASHED:
        line(image, rect.tl, rect.tr, color, line_style=line_style)
        line(image, rect.bl, rect.br, color, line_style=line_style)
        line(image, rect.tl, rect.bl, color, line_style=line_style)
        line(image, rect.tr, rect.br, color, line_style=line_style)
    else:
        raise ValueError(f"unknown line style: {line_style}")


def put_text(
    image: np.ndarray,
    text: str,
    origin: CVPoint,
    color: CVColor,
    thickness: int = 1,
    scale: float = 1,
):
    """
    Put `text` on `image` at `origin`.

    :param image: The image to draw the text
    :param text: The text to be drawn
    :param origin: The origin to start the text. The bottom of the first character
                   is set in the origin.
    :param color: The color of the text
    :param thickness: The thickness of the text
    :param scale: The scale of the text.
    """
    _warn_point_outside_image(image, origin)
    color = _ensure_color_int(color)
    cv.putText(
        image,
        text,
        (*map(int, origin),),
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness=thickness,
    )


def draw_contour(image: np.ndarray, contour: Contour, color: CVColor, thickness=1):
    """
    Draw a contour on an image.

    :param image: Image to draw on
    :param contour: Contour to draw
    :param color: Color to draw
    :param thickness: Thickness to draw with
    """
    color = _ensure_color_int(color)
    cv.drawContours(image, [contour.points], 0, color, thickness)


def draw_contours(
    image: np.ndarray, contours: Sequence[Contour], color: CVColor, thickness=1
):
    """
    Draw multiple contours on an image

    :param image: Image to draw on
    :param contours: Contours to draw
    :param color: Color to draw with
    :param thickness: Thickness to draw with
    """
    color = _ensure_color_int(color)
    points = (*map(lambda x: x.points, contours),)
    cv.drawContours(image, points, -1, color, thickness)


def wait_key(delay: int) -> str:
    """
    Wait for a key event infinitely (if `delay` is 0) or `delay` amount of milliseconds.

    An alias for cv.waitKey(delay) & 0xFF. See cv.waitKey(delay) for further documentation.
    Comparison of the key pressed can be found by `ord(str)`. For example

    >>> if wait_key(0) == ord('q'): continue

    :param delay: Amount of milliseconds to wait, or 0 for infinitely.
    :return: The key pressed.
    """
    return cv.waitKey(delay) & 0xFF


def _warn_point_outside_image(image: np.ndarray, point: CVPoint):
    if not isinstance(point, Point):
        point = Point(*point)
    if (not 0 <= point.y <= image.shape[0]) or (not 0 <= point.x <= image.shape[1]):
        warnings.warn(
            f"Point {(point.x, point.y)} outside image of shape {image.shape}."
        )


def _warn_rect_outside_image(image: np.ndarray, rect: CVRect):
    if not isinstance(rect, Point):
        rect = Rect(*rect)
    boundary = Rect(0, 0, *image.shape[:2][::-1])
    if not (
        rect.tl in boundary
        or rect.tr in boundary
        or rect.bl in boundary
        or rect.br in boundary
    ):
        warnings.warn(f"{rect} outside image of shape {image.shape}.")
