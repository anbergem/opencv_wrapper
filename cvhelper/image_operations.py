import numpy as np
import cv2 as cv


def morph_open(img: np.ndarray, size: int, iterations=1) -> np.ndarray:
    return cv.morphologyEx(
        img,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (size, size)),
        iterations=iterations,
    )


def morph_close(img: np.ndarray, size: int, iterations=1) -> np.ndarray:
    return cv.morphologyEx(
        img,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (size, size)),
        iterations=iterations,
    )


def normalize(img: np.ndarray, min: int = 0, max: int = 255) -> np.ndarray:
    normalized = np.zeros_like(img)
    cv.normalize(img, normalized, max, min, cv.NORM_MINMAX)
    return normalized


def resize(img: np.ndarray, factor: int) -> np.ndarray:
    """
    Resize an image with the given factor. A factor of 2 gives an image of half the size.
    :param img: Image to resize
    :param factor: Shrink factor. A factor of 2 halves the image size.
    :return: A resized image.
    """
    return cv.resize(
        img, None, fx=1 / factor, fy=1 / factor, interpolation=cv.INTER_CUBIC
    )


def color_to_gray(img: np.ndarray) -> np.ndarray:
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)
