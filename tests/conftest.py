import pytest

from opencv_wrapper.model import Contour


@pytest.fixture
def points(mocker):
    return mocker.Mock()


@pytest.fixture
def contour(points):
    return Contour(points)


@pytest.fixture
def image(mocker):
    img = mocker.MagicMock()
    img.__len__.return_value = 1
    img.size.return_value = 1
    return img


@pytest.fixture
def gray_image(image):
    image.ndim = 2
    return image


@pytest.fixture
def color_image(image):
    image.ndim = 3
    return image


@pytest.fixture
def image_uint8(image, np_mock):
    image.dtype = np_mock.uint8
