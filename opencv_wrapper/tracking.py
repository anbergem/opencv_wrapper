import cv2 as cv
import numpy as np


def dense_optical_flow(
    prev_frame: np.ndarray,
    next_frame: np.ndarray,
    pyr_scale: float = 0.5,
    levels: int = 3,
    winsize: int = 11,
    iterations: int = 1,
    poly_n: int = 4,
    poly_sigma: float = 1.1,
    gaussian_window: bool = True,
    initial_flow: np.ndarray = None,
) -> np.ndarray:
    """
    Calculate the dense optical flow between two frames, using the
    Farnebäck method.

    For further documentation on the parameters, see OpenCV documentation
    for cv2.calcOpticalFlowFarnaback.

    :param prev_frame: The initial frame
    :param next_frame: The frame after `prev_frame`, with displacement.
    :return: An image with the shape
    """
    if prev_frame.shape != next_frame.shape:
        raise ValueError(
            f"prev_frame and next_frame must have same dimensions: {prev_frame.shape}, {next_frame.shape}"
        )
    flags = 0
    if initial_flow is not None:
        flags += cv.OPTFLOW_USE_INITIAL_FLOW
    if gaussian_window:
        flags += cv.OPTFLOW_FARNEBACK_GAUSSIAN

    return cv.calcOpticalFlowFarneback(
        prev_frame,
        next_frame,
        initial_flow,
        pyr_scale,
        levels,
        winsize,
        iterations,
        poly_n,
        poly_sigma,
        flags,
    )
