from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="opencv-wrapper",
    version="0.1.0",
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
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Typing :: Typed",
    ],
    keywords="OpenCV",
    install_requires=["numpy<=1.15.3", "opencv-python<4"],
    python_requires=">=3.7",
)
