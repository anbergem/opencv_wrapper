Changelog
=========

v0.2.0
------

New
~~~
- Support for Python 3.6
- Support for OpenCV 4
- VideoWriter
- Point properties `cartesian` and `polar`

Changes
~~~~~~~
- Change `Rect.aspoints` property to `Rect.cartesian_corners()` function.
- `canny` function now has `high_pass_size=3` as default argument.
- `threshold_adaptive` now has more specifiable parameters and docs.

Fix
~~~

