Changelog
=========

dev
---

**New**

- Add tests for `Rect`, `Point`
- Add `__radd__`, `__rsub__` and `__neg__` for Point.

**Changes**

- `Rect.__contains__` now acts as `Rect` in C++ with being inclusive on the
  top-left side, and exclusive on the bottom-right side.
- `Rect` must now have a positive width and height.


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
