import numpy as np
import pytest

from opencv_wrapper.model import Rect, Point


class TestPoint:
    @pytest.mark.parametrize(
        "point1, point2, expected",
        [
            (Point(5, 3), Point(1, 2), Point(6, 5)),
            (Point(1.2, 4.6), Point(3.1, 7.8), Point(4.3, 12.4)),
            (Point(1.2, 4.6), (3.1, 7.8), Point(4.3, 12.4)),
            ((1.2, 4.6), Point(3.1, 7.8), Point(4.3, 12.4)),
        ],
    )
    def test_add(self, point1, point2, expected):
        assert isinstance(point1 + point2, Point)
        assert (*(point1 + point2),) == pytest.approx((*expected,))

    @pytest.mark.parametrize(
        "point1, point2, expected",
        [
            (Point(5, 3), Point(1, 2), Point(4, 1)),
            (Point(1.2, 4.6), Point(3.1, 7.8), Point(-1.9, -3.2)),
            (Point(1.2, 4.6), (3.1, 7.8), Point(-1.9, -3.2)),
            ((1.2, 4.6), Point(3.1, 7.8), Point(-1.9, -3.2)),
        ],
    )
    def test_sub(self, point1, point2, expected):
        assert isinstance(point1 - point2, Point)
        assert (*(point1 - point2),) == pytest.approx((*expected,))

    @pytest.mark.parametrize(
        "point, expected", [(Point(3, 4), 5), (Point(-5, 9), np.sqrt(25 + 81))]
    )
    def test_norm(self, point, expected):
        assert point.norm == pytest.approx(expected)

    @pytest.mark.parametrize(
        "point, expected",
        [
            (Point(1, 0), (1, 0)),
            (Point(0, 1), (1, np.pi / 2)),
            (Point(2, 2), (4 * np.sqrt(2) / 2, np.pi / 4)),
        ],
    )
    def test_polar(self, point, expected):
        assert point.polar() == pytest.approx(expected)


class TestRect:
    @pytest.mark.parametrize(
        "args, padding, expected",
        [
            ((2.5, 4, 2, 2.5), 0.5, Rect(2, 3.5, 3, 3.5)),
            ((2.5, 4, 2, 2.5), -0.5, Rect(3, 4.5, 1, 1.5)),
        ],
    )
    def test_padding_succeed(self, args, padding, expected):
        assert Rect(*args, padding=padding) == expected

    def test_padding_fail(self):
        with pytest.raises(ValueError):
            Rect(1, 1, 1, 1, padding=-1)

    @pytest.mark.parametrize(
        "rect, divisor, expected",
        [
            (Rect(0, 0, 0, 0), 4, Rect(0, 0, 0, 0)),
            (Rect(4, 4, 4, 4), 4, Rect(1, 1, 1, 1)),
            (Rect(4, 7, 2, 3), 2, Rect(2, 3.5, 1, 1.5)),
        ],
    )
    def test_div(self, rect, divisor, expected):
        assert rect / divisor == expected
        expected = Rect(*map(int, expected))
        assert rect // divisor == expected

    @pytest.mark.parametrize(
        "rect1, rect2, expected",
        [
            (Rect(0, 0, 5, 6), Rect(4, 4, 4, 3), Rect(0, 0, 8, 7)),
            (Rect(0, 0, 5, 6), Rect(1, 1, 2, 3), Rect(0, 0, 5, 6)),
            (Rect(0, 0, 5, 6), Rect(5, 6, 2, 3), Rect(0, 0, 7, 9)),
        ],
    )
    def test_operator_or(self, rect1, rect2, expected):
        assert rect1 | rect2 == expected

    @pytest.mark.parametrize(
        "rect1, rect2, expected",
        [
            (Rect(0, 0, 5, 6), Rect(4, 4, 4, 3), Rect(4, 4, 1, 2)),
            (Rect(0, 0, 5, 6), Rect(1, 1, 2, 3), Rect(1, 1, 2, 3)),
            (Rect(0, 0, 5, 6), Rect(5, 6, 2, 3), None),
        ],
    )
    def test_operator_and(self, rect1, rect2, expected):
        assert rect1 & rect2 == expected

    @pytest.mark.parametrize("other", [1, 1.2, "as", True, Point(3, 4)])
    def test_operator_or_non_rect(self, other):
        with pytest.raises(TypeError):
            Rect(0, 0, 1, 1) | other

    @pytest.mark.parametrize("other", [1, 1.2, "as", True, Point(3, 4)])
    def test_operator_and_non_rect(self, other):
        with pytest.raises(TypeError):
            Rect(0, 0, 1, 1) & other

    @pytest.mark.parametrize(
        "rect, point, expected",
        [
            (Rect(0, 0, 2, 2), Point(0, 0), True),
            (Rect(0, 0, 2, 2), Point(0, 1), True),
            (Rect(0, 0, 2, 2), Point(1, 0), True),
            (Rect(0, 0, 2, 2), Point(1, 1), True),
            (Rect(0, 0, 2, 2), Point(2, 0), False),
            (Rect(0, 0, 2, 2), Point(0, 2), False),
            (Rect(0, 0, 2, 2), Point(2, 2), False),
        ],
    )
    def test_contains(self, rect, point, expected):
        assert (point in rect) is expected


class TestContour:
    @pytest.fixture
    def cv_mock(self, mocker):
        return mocker.patch("opencv_wrapper.model.cv")

    @pytest.fixture
    def contour(self, mocker):
        points = mocker.Mock()
        return Contour(points)

    def test_bounding_rect(self, cv_mock, contour):
        cv_mock.boundingRect.return_value = 0, 0, 0, 0
        contour.bounding_rect
        cv_mock.boundingRect.assert_called_once_with(contour.points)

        # Test caching
        contour.bounding_rect
        cv_mock.boundingRect.assert_called_once()

    def test_moments(self, mocker, cv_mock, contour):
        cv_mock.moments.return_value = mocker.MagicMock()
        contour.area
        cv_mock.moments.assert_called_once()

        # Test caching
        contour.area
        cv_mock.moments.assert_called_once()
