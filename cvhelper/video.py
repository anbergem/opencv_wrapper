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
from typing import Optional

import cv2 as cv


@contextmanager
def load_video(filename: str):
    video = cv.VideoCapture(filename)
    if not video.isOpened():
        raise ValueError(f"Could not open video with filename {filename}")
    try:
        yield video
    finally:
        video.release()


def read_frames(
    video: cv.VideoCapture, from_frame: int = 0, to_frame: Optional[int] = None
):
    """
    :param video: Video object to read from.
    :param from_frame: Frame number to skip to.
    :param to_frame: Frame number to stop reading, exclusive.
    """
    if to_frame is not None and from_frame >= to_frame:
        raise ValueError(
            f"from_frame ({from_frame}) must be less than to_frame ({to_frame})"
        )

    ok, current = video.read()
    if not ok:
        raise ValueError(f"Could not read video.")

    next_ok, next = video.read()

    # Skip frames until from_frame
    counter = 0
    while from_frame > counter:
        if not next_ok:
            raise ValueError(f"Not enough frames to skip to frame {current}")
        current = next

        next_ok, next = video.read()
        counter += 1

    yield current

    # If next frame is also good
    if next_ok:
        current = next

        while True:
            # +1 to make to_frame exclusive
            if counter + 1 == to_frame:
                break

            next_ok, next = video.read()
            counter += 1

            yield current

            current = next

            if not next_ok:
                break
