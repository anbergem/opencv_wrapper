from typing import Union, Optional

import cv2 as cv
import numpy as np

from .model import Point, Rect


def norm(input: Union[Point, np.ndarray]) -> float:
    """
    Calculates the absolute L2 norm of the point or array.
    :param input: The n-dimensional point
    :return: The L2 norm of the n-dimensional point
    """
    if isinstance(input, Point):
        return cv.norm((*input,))
    else:
        return cv.norm(input)


def line_iterator(image: np.ndarray, p1: Point, p2: Point) -> np.ndarray:
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points.

    Credit: https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator

    :param image: The image being processed
    :param p1: The first point
    :param p2: The second point
    :return: An array that consists of the coordinates and intensities of each pixel on the line.
             (shape: [numPixels, 3(5)], row = [x,y, intensity(b, g, r)]), for gray-scale(bgr) image.
    """
    # define local variables for readability
    imageH = image.shape[0]
    imageW = image.shape[1]
    # P1X = P1[0]
    # P1.y = P1[1]
    # P2X = P2[0]
    # P2.y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = p2.x - p1.x
    dY = p2.y - p1.y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    color_chls = 1 if image.ndim == 2 else 3
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 2 + color_chls), dtype=np.int32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = p1.y > p2.y
    negX = p1.x > p2.x
    if p1.x == p2.x:  # vertical line segment
        itbuffer[:, 0] = p1.x
        if negY:
            itbuffer[:, 1] = np.arange(p1.y - 1, p1.y - dYa - 1, -1, dtype=np.int32)
        else:
            itbuffer[:, 1] = np.arange(p1.y + 1, p1.y + dYa + 1, dtype=np.int32)
    elif p1.y == p2.y:  # horizontal line segment
        itbuffer[:, 1] = p1.y
        if negX:
            itbuffer[:, 0] = np.arange(p1.x - 1, p1.x - dXa - 1, -1, dtype=np.int32)
        else:
            itbuffer[:, 0] = np.arange(p1.x + 1, p1.x + dXa + 1, dtype=np.int32)
    else:  # diagonal line segment
        # TODO: error here when drawing from bottom right to top left diagonal.
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX / dY
            if negY:
                itbuffer[:, 1] = np.arange(p1.y - 1, p1.y - dYa - 1, -1, dtype=np.int32)
            else:
                itbuffer[:, 1] = np.arange(p1.y + 1, p1.y + dYa + 1, dtype=np.int32)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - p1.y)).astype(np.int) + p1.x
        else:
            slope = dY / dX
            if negX:
                itbuffer[:, 0] = np.arange(p1.x - 1, p1.x - dXa - 1, -1, dtype=np.int32)
            else:
                itbuffer[:, 0] = np.arange(p1.x + 1, p1.x + dXa + 1, dtype=np.int32)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - p1.x)).astype(np.int) + p1.y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    # Get three values if color image
    num_channels = 2 if image.ndim == 2 else slice(2, None, None)
    itbuffer[:, num_channels] = image[itbuffer[:, 1], itbuffer[:, 0]]

    return itbuffer


def rect_intersection(rect1: Rect, rect2: Rect) -> Optional[Rect]:
    """
    Calculate the intersection between two rectangles.

    :param rect1: First rectangle
    :param rect2: Second rectangle
    :return: A rectangle representing the intersection between `rect1` and `rect2`
             if it exists, else None.
    """
    top = min(rect1, rect2, key=lambda x: x.tl.y)
    bottom = max(rect2, rect1, key=lambda x: x.tl.y)
    if top.br.y < bottom.tl.y or top.br.x < bottom.bl.x or top.bl.x > bottom.br.x:
        return None

    tl = Point(max(top.tl.x, bottom.tl.x), bottom.tl.y)

    width = min(bottom.br.x, top.br.x) - tl.x
    height = min(bottom.br.y, top.br.y) - tl.y

    return Rect(tl.x, tl.y, width, height)
