import enum
from typing import Tuple, Optional

import cv2 as cv
import numpy as np

from .model import Rect, Point, Contour


class MorphShape(enum.Enum):
    RECT: int = cv.MORPH_RECT
    CROSS: int = cv.MORPH_CROSS
    CIRCLE: int = cv.MORPH_ELLIPSE


class AngleUnit(enum.Enum):
    RADIANS = enum.auto()
    DEGREES = enum.auto()


def find_external_contours(image: np.ndarray) -> Tuple[Contour]:
    """
    Find the external contours in the `image`.

    Alias for cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    :param image: The image in with to find the contours
    :return: A tuple of Contour objects
    """
    _, contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return (*map(Contour, contours),)


def dilate(
    image: np.ndarray,
    kernel_size,
    shape: MorphShape = MorphShape.RECT,
    iterations: int = 1,
):
    _error_if_image_empty(image)
    return cv.dilate(
        image,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def erode(
    image: np.ndarray,
    kernel_size,
    shape: MorphShape = MorphShape.RECT,
    iterations: int = 1,
):
    _error_if_image_empty(image)
    return cv.erode(
        image,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def morph_open(
    image: np.ndarray,
    kernel_size: int,
    shape: MorphShape = MorphShape.RECT,
    iterations=1,
) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.morphologyEx(
        image,
        cv.MORPH_OPEN,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def morph_close(
    image: np.ndarray,
    kernel_size: int,
    shape: MorphShape = MorphShape.RECT,
    iterations=1,
) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.morphologyEx(
        image,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(shape.value, (kernel_size, kernel_size)),
        iterations=iterations,
    )


def normalize(image: np.ndarray, min: int = 0, max: int = 255) -> np.ndarray:
    _error_if_image_empty(image)
    normalized = np.zeros_like(image)
    cv.normalize(image, normalized, max, min, cv.NORM_MINMAX)
    return normalized


def resize(
    image: np.ndarray,
    factor: Optional[int] = None,
    shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Resize an image with the given factor. A factor of 2 gives an image of half the size.

    If the image has 4 dimensions, it is assumed to be a series of images.
    :param image: Image to resize
    :param factor: Shrink factor. A factor of 2 halves the image size.
    :param shape: Output image size.
    :return: A resized image
    """
    if shape is None and factor is None:
        raise ValueError("Either shape or factor must be specified.")
    _error_if_image_empty(image)
    if image.ndim == 2 or image.ndim == 3:
        if shape is not None:
            return cv.resize(image, shape, interpolation=cv.INTER_CUBIC)
        else:
            return cv.resize(
                image, None, fx=1 / factor, fy=1 / factor, interpolation=cv.INTER_CUBIC
            )
    raise ValueError("Image must have either 2 or 3 dimensions.")


def gray2bgr(image: np.ndarray) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.cvtColor(image, cv.COLOR_GRAY2BGR)


def bgr2gray(image: np.ndarray) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def bgr2hsv(image: np.ndarray) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def bgr2xyz(image: np.ndarray) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.cvtColor(image, cv.COLOR_BGR2XYZ)


def bgr2hls(image: np.ndarray) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.cvtColor(image, cv.COLOR_BGR2HLS)


def bgr2luv(image: np.ndarray) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.cvtColor(image, cv.COLOR_BGR2LUV)


def blur_gaussian(
    image: np.ndarray, kernel_size: int = 3, sigma_x=None, sigma_y=None
) -> np.ndarray:
    _error_if_image_empty(image)
    if sigma_x is None:
        sigma_x = 0
    if sigma_y is None:
        sigma_y = 0

    return cv.GaussianBlur(
        image, ksize=(kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y
    )


def blur_median(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.medianBlur(image, kernel_size)


def threshold_adaptive(image: np.ndarray, block_size: int, c: int = 0) -> np.ndarray:
    _error_if_image_empty(image)
    return cv.adaptiveThreshold(
        image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, c
    )


def threshold_otsu(
    image: np.ndarray, max_value: int = 255, inverse: bool = False
) -> np.ndarray:
    _error_if_image_empty(image)
    flags = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    flags += cv.THRESH_OTSU
    _, img = cv.threshold(image, 0, max_value, flags)
    return img


def threshold_binary(
    image: np.ndarray, value: int, max_value: int = 255, inverse: bool = False
) -> np.ndarray:
    _error_if_image_empty(image)
    flags = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    _, img = cv.threshold(image, value, max_value, flags)
    return img


def threshold_tozero(image: np.ndarray, value: int, max_value: int = 255) -> np.ndarray:
    _error_if_image_empty(image)
    _, img = cv.threshold(image, value, max_value, cv.THRESH_TOZERO)
    return img


def threshold_otsu_tozero(image: np.ndarray, max_value: int = 255) -> np.ndarray:
    _error_if_image_empty(image)
    _, img = cv.threshold(image, 0, max_value, cv.THRESH_OTSU | cv.THRESH_TOZERO)
    return img


def canny(
    image: np.ndarray,
    low_threshold: float,
    high_threshold: float,
    high_pass_size: int,
    l2_gradient=True,
) -> np.ndarray:
    """
    Perform Canny's edge detection on `image`.
    :param image: The image to be processed.
    :param low_threshold: The lower threshold in the hysteresis thresholding.
    :param high_threshold: The higher threshold in the hysteresis thresholding.
    :param high_pass_size: The size of the Sobel filter, used to find gradients.
    :param l2_gradient: Whether to use the L2 gradient. The L1 gradient is used if false.
    :return: Binary image of thinned edges.
    """
    _error_if_image_empty(image)
    if high_pass_size not in [3, 5, 7]:
        raise ValueError(f"High pass size must be either 3, 5 or 7: {high_pass_size}")
    return cv.Canny(
        image,
        threshold1=low_threshold,
        threshold2=high_threshold,
        apertureSize=high_pass_size,
        L2gradient=l2_gradient,
    )


def scale_contour_to_rect(contour: Contour, rect: Rect) -> Contour:
    contour = Contour(contour.points)
    for i in range(len(contour)):
        contour[i, 0] = contour[i, 0] - rect.x
        contour[i, 1] = contour[i, 1] - rect.y

    return contour


def rotate_image(
    image: np.ndarray,
    center: Point,
    angle: float,
    scale: int = 1,
    unit: AngleUnit = AngleUnit.RADIANS,
) -> np.ndarray:
    if unit is AngleUnit.RADIANS:
        angle = 180 / np.pi * angle
    rotation_matrix = cv.getRotationMatrix2D((*center,), angle, scale=scale)

    if image.ndim == 2:
        return cv.warpAffine(image, rotation_matrix, image.shape[::-1])
    elif image.ndim == 3:
        copy = np.zeros_like(image)
        shape = image.shape[-2::-1]  # The two first, reversed
        for i in range(copy.shape[-1]):
            copy[..., i] = cv.warpAffine(image[..., i], rotation_matrix, shape)
        return copy
    else:
        raise ValueError("Image must have 2 or 3 dimensions.")


def _error_if_image_empty(image: np.ndarray) -> None:
    if image is None or image.size == 0:
        raise ValueError("Image is empty")
