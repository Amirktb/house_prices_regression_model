#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path

from setuptools import find_packages, setup

# meta-data
NAME = 'PricePrediction_regression_model'
DESCRIPTION = "Rgression model package for price predictin of house prices."
URL = "https://github.com/Amirktb/house_prices_regression_model"
EMAIL = "kotobiamir@gmail.com"
AUTHOR = "Amirktb"
REQUIRES_PYTHON = ">=3.6.0"

long_description = DESCRIPTION

# package's version
about = {}
ROOT_DIR = Path(__file__).resolve().parent
REQUIREMENTS_DIR = ROOT_DIR / 'requirements'
PACKAGE_DIR = ROOT_DIR / 'regression_model'
with open(PACKAGE_DIR / "VERSION") as f:
    _version = f.read().strip()
    about["__version__"] = _version

def list_reqs(fname="requirements.txt"):
    with open(REQUIREMENTS_DIR / fname) as fd:
        return fd.read().splitlines()
    
setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    package_data={"regression_model": ["VERSION"]},
    install_requires=list_reqs(),
    extras_require={},
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
)