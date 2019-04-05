import os.path
import sys

from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

requirements = ["numpy<=1.15.3", "opencv-python<=4.0.1"]

if sys.version_info[1] == 6:
    requirements.append("dataclasses")

here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, "opencv_wrapper", "__version__.py"), "r") as f:
    exec(f.read(), about)


setup(
    name=about["__title__"],
    version=about["__version__"],
    author=about["__author__"],
    author_email=about["__author_email__"],
    description=about["__description__"],
    license=about["__license__"],
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=about["__url__"],
    packages=["opencv_wrapper"],
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Typing :: Typed",
    ],
    keywords="OpenCV",
    install_requires=requirements,
    python_requires=">=3.6",
)
