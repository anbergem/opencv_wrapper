import sys

from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

requirements = ["numpy<=1.15.3", "opencv-python<=4.0.1"]

if sys.version_info[1] == 6:
    requirements.append("dataclasses")

setup(
    name="opencv-wrapper",
    version="0.2.0",
    packages=["opencv_wrapper"],
    author="Andreas Bergem",
    author_email="bergem.andreas@gmail.com",
    description="A python wrapper for OpenCV",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbergem/opencv_wrapper",
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
