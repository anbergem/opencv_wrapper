"""
Convenience functions for reading videos.

Usage:

>>> import cv2 as cv
>>> with load_video("path/to/file") as video:
>>>    for frame in read_frames(video):
>>>        cv.imshow("Frame", frame)
>>>        cv.waitKey(1)
"""
from contextlib import contextmanager
from typing import Optional, Iterator

import cv2 as cv
import numpy as np


@contextmanager
def load_video(filename: str):
    """
    Open the video file
    :param filename:
    """
    video = cv.VideoCapture(filename)
    if not video.isOpened():
        raise ValueError(f"Could not open video with filename {filename}")
    try:
        yield video
    finally:
        video.release()


def read_frames(
    video: cv.VideoCapture, start: int = 0, stop: Optional[int] = None, step: int = 1
) -> Iterator[np.ndarray]:
    """
    :param video: Video object to read from.
    :param start: Frame number to skip to.
    :param stop: Frame number to stop reading, exclusive.
    :param step: Step to iterate over frames. Similar to range's step. Must be greater than 0.
    """
    if stop is not None and start >= stop:
        raise ValueError(f"from_frame ({start}) must be less than to_frame ({stop})")
    if step <= 0 or not isinstance(step, int):
        raise ValueError(f"Step must be an integer greater than 0: {step}")

    ok, current = video.read()
    if not ok:
        raise ValueError(f"Could not read video.")

    next_ok, next = video.read()

    # Skip frames until from_frame
    counter = 0
    while start > counter:
        if not next_ok:
            raise ValueError(
                f"Not enough frames to skip to frame {start}. File ended at frame {counter}."
            )
        current = next

        next_ok, next = video.read()
        counter += 1

    yield current

    # If next frame is also good
    if next_ok:
        current = next

        while True:
            for i in range(step):
                # +1 to make to_frame exclusive
                if counter + 1 == stop:
                    return

                next_ok, next = video.read()
                counter += 1

                if not next_ok:
                    if i == step - 1:
                        yield current
                    return

            yield current

            current = next

            if not next_ok:
                return
