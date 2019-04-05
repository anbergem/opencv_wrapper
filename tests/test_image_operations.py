import pytest

from opencv_wrapper.image_operations import find_external_contours


@pytest.fixture
def cv_mock(mocker):
    return mocker.patch("opencv_wrapper.image_operations.cv")


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
