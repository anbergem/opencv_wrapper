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
    return mocker.MagicMock()
