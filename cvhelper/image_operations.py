import numpy as np
import cv2 as cv


def morph_open(image: np.ndarray, size: int, iterations=1) -> np.ndarray:
    return cv.morphologyEx(
        image,
        cv.MORPH_OPEN,
        cv.getStructuringElement(cv.MORPH_RECT, (size, size)),
        iterations=iterations,
    )


def morph_close(image: np.ndarray, size: int, iterations=1) -> np.ndarray:
    return cv.morphologyEx(
        image,
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_RECT, (size, size)),
        iterations=iterations,
    )


def normalize(image: np.ndarray, min: int = 0, max: int = 255) -> np.ndarray:
    normalized = np.zeros_like(image)
    cv.normalize(image, normalized, max, min, cv.NORM_MINMAX)
    return normalized


def resize(image: np.ndarray, factor: int) -> np.ndarray:
    """
    Resize an image with the given factor. A factor of 2 gives an image of half the size.
    :param image: Image to resize
    :param factor: Shrink factor. A factor of 2 halves the image size.
    :return: A resized image.
    """
    return cv.resize(
        image, None, fx=1 / factor, fy=1 / factor, interpolation=cv.INTER_CUBIC
    )


def color_to_gray(image: np.ndarray) -> np.ndarray:
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)


def blur_gaussian(
    image: np.ndarray, kernel_size: int = 3, sigma_x=None, sigma_y=None
) -> np.ndarray:
    if sigma_x is None:
        sigma_x = 0
    if sigma_y is None:
        sigma_y = 0

    return cv.GaussianBlur(
        image, ksize=(kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y
    )


def threshold_otsu(image: np.ndarray, max_value: int = 255) -> np.ndarray:
    _, img = cv.threshold(image, 0, max_value, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return img


def threshold_binary(image: np.ndarray, value: int, max_value: int = 255) -> np.ndarray:
    _, img = cv.threshold(image, value, max_value, cv.THRESH_BINARY)
    return img


def threshold_tozero(image: np.ndarray, value: int, max_value: int = 255) -> np.ndarray:
    _, img = cv.threshold(image, value, max_value, cv.THRESH_TOZERO)
    return img
