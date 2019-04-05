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
from typing import Optional, Iterator, Any

import cv2 as cv
import numpy as np

from .image_operations import _error_if_image_empty


@contextmanager
def load_camera(index: int = 0) -> Iterator[Any]:
    """Open a camera for video capturing.

    :param index: Index of the camera to open.

                  For more details see `cv2.VideoCapture(index) documentation`_

    .. _cv2.VideoCapture(index) documentation : https://docs.opencv.org/3.4.5/d8/dfe/classcv_1_1VideoCapture.html#a5d5f5dacb77bbebdcbfb341e3d4355c1
    """
    video = cv.VideoCapture(index)
    if not video.isOpened():
        raise ValueError(f"Could not open camera with index {index}")
    try:
        yield video
    finally:
        video.release()


@contextmanager
def load_video(filename: str) -> Iterator[Any]:
    """
    Open a video file

    :param filename: It an be:

                     * Name of video file
                     * An image sequence
                     * A URL of a video stream

                     For more details see `cv2.VideoCapture(filename) documentation`_

    .. _cv2.VideoCapture(filename) documentation: https://docs.opencv.org/3.4.3/d8/dfe/classcv_1_1VideoCapture.html#a85b55cf6a4a50451367ba96b65218ba1
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
    """Read frames of a video object.

    Start, stop and step work as built-in range.

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


class VideoWriter(object):
    """
    A video writer for writing videos, using OpenCV's `cv.VideoWriter`.

    The video writer is lazy, in that it waits to receive the first frame, before determining
    the frame size for the video writer. This is in contrast to OpenCV's video writer, which
    expects a frame size up front.
    """

    def __init__(
        self, filename: str, fps: int = None, capture: Any = None, fourcc: str = "MJPG"
    ):
        """
        Either `fps` or `capture` must be provided.
        
        For additional documentation, see `cv2.VideoWriter documentation`_

        .. _cv2.VideoWriter documentation: https://docs.opencv.org/3.4.5/dd/d9e/classcv_1_1VideoWriter.html

        :param filename: Name of the output video file.
        :param fps: Framerate of the created video stream.
        :param capture: A capture object from cv.VideoCapture or :func:`load_video`. Used to retrieve
                        fps if `fps` is not provided.
        :param fourcc: 4-character code of codec used to compress the frames. See
                       `documentation <https://docs.opencv.org/3.4.5/dd/d9e/classcv_1_1VideoWriter.html#ac3478f6257454209fa99249cc03a5c59>`_
        """
        self.filename = filename
        self.fourcc = fourcc

        if fps is not None:
            self.fps = fps
        elif capture is not None:
            self.fps = capture.get(cv.CAP_PROP_FPS)
        else:
            raise ValueError("Either `fps` or `capture` must be provided")

        self.writer = None
        self.frame_shape = None

    def write(self, frame):
        """Write a frame to the video.

        The frame must be the same size each time the frame is written.

        :param frame: Image to be written
        """
        _error_if_image_empty(frame)
        if self.writer is None:
            self.frame_shape = frame.shape
            self.writer = cv.VideoWriter()
            self.writer.open(
                self.filename,
                cv.VideoWriter_fourcc(*self.fourcc),
                self.fps,
                (frame.shape[1], frame.shape[0]),
                frame.ndim == 3,
            )
        else:
            if frame.shape != self.frame_shape:
                raise ValueError(
                    f"frame.shape {frame.shape} does not match previous shape {self.frame_shape}"
                )
        # Write to video
        self.writer.write(frame)
