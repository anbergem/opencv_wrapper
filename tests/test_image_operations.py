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
    normalize,
    resize,
    blur_gaussian,
    blur_median,
    threshold_otsu,
    threshold_binary,
    threshold_tozero,
    threshold_adaptive,
)

functions = inspect.getmembers(opencv_wrapper.image_operations, inspect.isfunction)


@pytest.fixture
def cv_mock(mocker):
    mock = mocker.patch("opencv_wrapper.image_operations.cv")
    mock.threshold.return_value = (None, None)
    return mock


@pytest.fixture
def np_mock(mocker):
    return mocker.patch("opencv_wrapper.image_operations.np")


@pytest.fixture
def _empty_func(mocker):
    return mocker.patch("opencv_wrapper.image_operations._error_if_image_empty")


@pytest.fixture
def _error_not_color(mocker):
    return mocker.patch("opencv_wrapper.image_operations._error_if_image_not_color")


@pytest.fixture
def _error_not_gray(mocker):
    return mocker.patch("opencv_wrapper.image_operations._error_if_image_not_gray")


@pytest.fixture
def _wrong_dtype_func(mocker):
    return mocker.patch("opencv_wrapper.image_operations._error_if_image_wrong_dtype")


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


def test_normalize(mocker, cv_mock, image, _empty_func):
    min = mocker.Mock()
    max = mocker.Mock()

    normalize(image, min, max)

    _empty_func.assert_called_once_with(image)

    cv_mock.normalize.assert_called_once()
    args = cv_mock.normalize.call_args[0]

    assert args[0] is image
    assert args[2] is max
    assert args[3] is min
    assert args[4] is cv_mock.NORM_MINMAX


@pytest.fixture
def shape(mocker):
    return mocker.Mock()


def test_resize_shape(shape, cv_mock, gray_image, color_image, _empty_func):
    for image in gray_image, color_image:
        resize(image, shape=shape)

        _empty_func.assert_called_once_with(image)
        cv_mock.resize.assert_called_once_with(
            image, shape, interpolation=cv_mock.INTER_CUBIC
        )

        _empty_func.reset_mock()
        cv_mock.reset_mock()


def test_resize_factor(mocker, cv_mock, gray_image, color_image, _empty_func):
    factor = mocker.MagicMock()

    for image in gray_image, color_image:
        resize(image, factor=factor)

        _empty_func.assert_called_once_with(image)
        cv_mock.resize.assert_called_once_with(
            image, None, fx=1 / factor, fy=1 / factor, interpolation=cv_mock.INTER_CUBIC
        )

        _empty_func.reset_mock()
        cv_mock.reset_mock()


@pytest.mark.parametrize(
    "kwargs", [{}, dict(shape=None), dict(factor=None), dict(shape=None, factor=None)]
)
def test_resize_fail_no_shape_or_factor(gray_image, kwargs):
    with pytest.raises(ValueError):
        resize(gray_image, **kwargs)


def test_resize_fail_image_wrong_dim(image, shape):
    for ndim in 1, 4:
        image.ndim = ndim
        with pytest.raises(ValueError):
            resize(image, shape=shape)


@pytest.mark.parametrize(
    "name, function",
    [(*f,) for f in functions if "2" in f[0] and not f[0].startswith("gray")],
)
def test_color2other(
    cv_mock, color_image, name, function, _empty_func, _error_not_color
):
    function(color_image)

    _empty_func.assert_called_once()
    _error_not_color.assert_called_once()

    # For example cv.COLOR_BGR2GRAY for function bgr2gray.
    type = getattr(cv_mock, f"COLOR_{name.upper()}")
    cv_mock.cvtColor.assert_called_once_with(color_image, type)


@pytest.mark.parametrize(
    "name, function",
    [(*f,) for f in functions if "2" in f[0] and f[0].startswith("gray")],
)
def test_color2other(cv_mock, gray_image, name, function, _empty_func, _error_not_gray):
    function(gray_image)

    _empty_func.assert_called_once()
    _error_not_gray.assert_called_once()

    # For example cv.COLOR_BGR2GRAY for function bgr2gray.
    type = getattr(cv_mock, f"COLOR_{name.upper()}")
    cv_mock.cvtColor.assert_called_once_with(gray_image, type)


@pytest.fixture
def kernel_size(mocker):
    return mocker.Mock()


def test_blur_gaussian_with_sigma(image, kernel_size, mocker, cv_mock, _empty_func):
    sigma_x, sigma_y = mocker.Mock(), mocker.Mock()

    blur_gaussian(image, kernel_size, sigma_x, sigma_y)

    _empty_func.assert_called_once()
    cv_mock.GaussianBlur.assert_called_once_with(
        image, ksize=(kernel_size, kernel_size), sigmaX=sigma_x, sigmaY=sigma_y
    )


def test_blur_gaussian_without_sigma(image, kernel_size, cv_mock, _empty_func):
    blur_gaussian(image, kernel_size)

    _empty_func.assert_called_once()
    cv_mock.GaussianBlur.assert_called_once_with(
        image, ksize=(kernel_size, kernel_size), sigmaX=0, sigmaY=0
    )


def test_blur_median(image, kernel_size, cv_mock, _empty_func):
    blur_median(image, kernel_size)

    _empty_func.assert_called_once()
    cv_mock.medianBlur.assert_called_once_with(image, kernel_size)


def test_threshold_binary(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    value = mocker.Mock()
    max_value = mocker.Mock()

    threshold_binary(image_uint8, value, max_value)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(
        image_uint8, [np_mock.float32, np_mock.uint8]
    )

    flags = cv_mock.THRESH_BINARY
    cv_mock.threshold.assert_called_once_with(image_uint8, value, max_value, flags)


def test_threshold_binary_inverse(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    value = mocker.Mock()
    max_value = mocker.Mock()

    threshold_binary(image_uint8, value, max_value, inverse=True)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(
        image_uint8, [np_mock.float32, np_mock.uint8]
    )

    flags = cv_mock.THRESH_BINARY_INV
    cv_mock.threshold.assert_called_once_with(image_uint8, value, max_value, flags)


def test_threshold_tozero(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    value = mocker.Mock()
    max_value = mocker.Mock()

    threshold_tozero(image_uint8, value, max_value)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(
        image_uint8, [np_mock.float32, np_mock.uint8]
    )

    flags = cv_mock.THRESH_TOZERO
    cv_mock.threshold.assert_called_once_with(image_uint8, value, max_value, flags)


def test_threshold_tozero_inverse(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    value = mocker.Mock()
    max_value = mocker.Mock()

    threshold_tozero(image_uint8, value, max_value, inverse=True)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(
        image_uint8, [np_mock.float32, np_mock.uint8]
    )

    flags = cv_mock.THRESH_TOZERO_INV
    cv_mock.threshold.assert_called_once_with(image_uint8, value, max_value, flags)


def test_threshold_otsu(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    max_value = mocker.Mock()

    threshold_otsu(image_uint8, max_value)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(
        image_uint8, [np_mock.float32, np_mock.uint8]
    )

    flags = cv_mock.THRESH_BINARY
    # implementation uses iadd, not add, has to be same when mocking
    flags += cv_mock.THRESH_OTSU
    cv_mock.threshold.assert_called_once_with(image_uint8, 0, max_value, flags)


def test_threshold_otsu_inverse(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    max_value = mocker.Mock()

    threshold_otsu(image_uint8, max_value, inverse=True)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(
        image_uint8, [np_mock.float32, np_mock.uint8]
    )

    flags = cv_mock.THRESH_BINARY_INV
    # implementation uses iadd, not add, has to be same when mocking
    flags += cv_mock.THRESH_OTSU
    cv_mock.threshold.assert_called_once_with(image_uint8, 0, max_value, flags)


def test_adaptive_threshold(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    block_size = mocker.Mock()
    c = mocker.Mock()

    threshold_adaptive(image_uint8, block_size, c, inverse=True)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(image_uint8, [np_mock.uint8])

    method = cv_mock.ADAPTIVE_THRESH_GAUSSIAN_C
    # implementation uses iadd, not add, has to be same when mocking
    flags = cv_mock.THRESH_BINARY_INV
    cv_mock.adaptiveThreshold.assert_called_once_with(
        image_uint8, 255, method, flags, block_size, c
    )


def test_adaptive_threshold2(
    mocker, image_uint8, cv_mock, np_mock, _empty_func, _wrong_dtype_func
):
    block_size = mocker.Mock()
    c = mocker.Mock()

    threshold_adaptive(image_uint8, block_size, c, weighted=False)

    _empty_func.assert_called_once_with(image_uint8)
    _wrong_dtype_func.assert_called_once_with(image_uint8, [np_mock.uint8])

    method = cv_mock.ADAPTIVE_THRESH_MEAN_C
    # implementation uses iadd, not add, has to be same when mocking
    flags = cv_mock.THRESH_BINARY
    cv_mock.adaptiveThreshold.assert_called_once_with(
        image_uint8, 255, method, flags, block_size, c
    )
