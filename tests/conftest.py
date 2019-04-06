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
