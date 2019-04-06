import inspect

import pytest

import opencv_wrapper.image_operations
from opencv_wrapper.image_operations import (
    MorphShape,
    find_external_contours,
    dilate,
    erode,
    morph_open,
    morph_close,
)

functions = inspect.getmembers(opencv_wrapper.image_operations, inspect.isfunction)


@pytest.fixture
def cv_mock(mocker):
    return mocker.patch("opencv_wrapper.image_operations.cv")


@pytest.fixture
def _empty_func(mocker):
    return mocker.patch("opencv_wrapper.image_operations._error_if_image_empty")


def test_bounding_rect_opencv3x(cv_mock, image, contour, points):
    cv_mock.findContours.return_value = image, [points], None
    ret = find_external_contours(image)
    cv_mock.findContours.assert_called_once_with(
        image, cv_mock.RETR_EXTERNAL, cv_mock.CHAIN_APPROX_SIMPLE
    )

    assert ret[0] == contour


def test_bounding_rect_opencv4x(cv_mock, image, contour, points):
    cv_mock.findContours.return_value = [points], None
    ret = find_external_contours(image)
    cv_mock.findContours.assert_called_once_with(
        image, cv_mock.RETR_EXTERNAL, cv_mock.CHAIN_APPROX_SIMPLE
    )

    assert ret[0] == contour


@pytest.mark.parametrize("function", [dilate, erode])
def test_erode_and_dilate(function, mocker, cv_mock, image, _empty_func):
    struct_mock = mocker.Mock()
    cv_mock.getStructuringElement.return_value = struct_mock
    kernel_size = 3
    shape = MorphShape.RECT
    iterations = 3

    function(image, kernel_size, shape, iterations)

    _empty_func.assert_called_once_with(image)
    cv_mock.getStructuringElement.assert_called_once_with(
        shape.value, (kernel_size, kernel_size)
    )
    if function is dilate:
        cv_mock.dilate.assert_called_once_with(
            image, struct_mock, iterations=iterations
        )
    elif function is erode:
        cv_mock.erode.assert_called_once_with(image, struct_mock, iterations=iterations)


@pytest.mark.parametrize("function", [morph_open, morph_close])
def test_morph_open_and_close(function, mocker, cv_mock, image, _empty_func):
    struct_mock = mocker.Mock()
    cv_mock.getStructuringElement.return_value = struct_mock
    kernel_size = 3
    shape = MorphShape.RECT
    iterations = 3

    function(image, kernel_size, shape, iterations)

    _empty_func.assert_called_once_with(image)
    cv_mock.getStructuringElement.assert_called_once_with(
        shape.value, (kernel_size, kernel_size)
    )
    if function is morph_open:
        op = cv_mock.MORPH_OPEN
    elif function is morph_close:
        op = cv_mock.MORPH_CLOSE
    else:
        assert 0

    cv_mock.morphologyEx.assert_called_with(
        image, op, struct_mock, iterations=iterations
    )
