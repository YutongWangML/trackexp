"""
Setup script for the trackexp package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="trackexp",
    version="0.1.0",
    author="Yutong Wang",
    author_email="ywang562@iit.edu",
    description="A lightweight experiment tracking module for Python",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YutongWangML/trackexp",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.7",
    install_requires=[
        "humanhash3",
        "pandas",
        "numpy"
    ],
)
