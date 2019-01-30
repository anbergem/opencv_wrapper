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


def read_frames(video: cv.VideoCapture):
    ok, frame = video.read()
    if not ok:
        raise ValueError(f"Could not read video.")

    next_ok, next = video.read()

    yield frame

    # If next frame is also good
    if next_ok:
        frame = next

        while True:
            next_ok, next = video.read()

            yield frame

            frame = next

            if not next_ok:
                break
