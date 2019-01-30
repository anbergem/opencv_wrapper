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
):
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
