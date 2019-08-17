Changelog
=========

dev
---

**New**

**Changes**

v0.2.3 (2019-08-17)
-------------------

**New**

- Add tests
- Add `__radd__`, `__rsub__` and `__neg__` for Point.
- Add `inverse` option for remaining threhsolding functions
- The video received from `load_video` and `load_camera` are now iterable

**Changes**

- `Rect.__contains__` now acts as `Rect` in C++ with being inclusive on the
  top-left side, and exclusive on the bottom-right side.
- `Rect` must now have a positive width and height.
- Resize parameters factor and shape changed to keyword only


v0.2.1 (2019-04-05)
-------------------

**Bugfix**
- Fix `find_external_contour` after inclusion of OpenCV 4


v0.2.0 (2019-04-05)
-------------------

**New**

- Support for Python 3.6
- Support for OpenCV 4
- VideoWriter
- Point properties `cartesian` and `polar`

**Changes**

- Change `Rect.aspoints` property to `Rect.cartesian_corners()` function.
- `canny` function now has `high_pass_size=3` as default argument.
- `threshold_adaptive` now has more specifiable parameters and docs.
