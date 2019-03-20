from setuptools import setup

with open("README.md") as fh:
    long_description = fh.read()

setup(
    name="cvhelper",
    version="0.0.4",
    packages=["cvhelper"],
    author="Andreas Bergem",
    author_email="bergem.andreas@gmail.com",
    description="A helper package for OpenCV",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/anbergem/cvhelper",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords="opencv",
    install_requires=["numpy<=1.15.3", "opencv-python<=3.4.3.18"],
    python_requires=">=3.7",
)
