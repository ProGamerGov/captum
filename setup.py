#!/usr/bin/env python3

# Welcome to the PyTorch Captum setup.py.
#
# Environment variables for feature toggles:
#
#   BUILD_INSIGHTS
#     enables Captum Insights build via yarn
#

import os
import re
import sys

from setuptools import find_packages, setup

REQUIRED_MAJOR = 3
REQUIRED_MINOR = 6

# Check for python version
if sys.version_info < (REQUIRED_MAJOR, REQUIRED_MINOR):
    error = (
        "Your version of python ({major}.{minor}) is too old. You need "
        "python >= {required_major}.{required_minor}."
    ).format(
        major=sys.version_info.major,
        minor=sys.version_info.minor,
        required_minor=REQUIRED_MINOR,
        required_major=REQUIRED_MAJOR,
    )
    sys.exit(error)


TUTORIALS_REQUIRES = ["umap-learn", "scikit-learn", "torchtext", "torchvision"]

TEST_REQUIRES = ["pytest", "pytest-cov", "parameterized"]

DEV_REQUIRES = (
    TUTORIALS_REQUIRES
    + TEST_REQUIRES
    + [
        "black==22.3.0",
        "flake8",
        "sphinx",
        "sphinx-autodoc-typehints",
        "sphinxcontrib-katex",
        "mypy>=0.760",
        "usort==0.6.4",
        "ufmt",
        "annoy",
    ]
)

# get version string from module
with open(os.path.join(os.path.dirname(__file__), "captum/__init__.py"), "r") as f:
    version = re.search(r"__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M).group(1)

# read in README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setup(
        name="captum",
        version=version,
        description="Model interpretability for PyTorch",
        author="PyTorch Team",
        license="BSD-3",
        url="https://captum.ai",
        project_urls={
            "Documentation": "https://captum.ai",
            "Source": "https://github.com/pytorch/captum",
            "conda": "https://anaconda.org/pytorch/captum",
        },
        keywords=[
            "Model Interpretability",
            "Model Understanding",
            "Feature Importance",
            "Neuron Importance",
            "PyTorch",
        ],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Developers",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: BSD License",
            "Programming Language :: Python :: 3 :: Only",
            "Topic :: Scientific/Engineering",
        ],
        long_description=long_description,
        long_description_content_type="text/markdown",
        python_requires=">=3.6",
        install_requires=["matplotlib", "numpy", "packaging", "torch>=1.6"],
        packages=find_packages(exclude=("tests", "tests.*")),
        extras_require={
            "dev": DEV_REQUIRES,
            "test": TEST_REQUIRES,
            "tutorials": TUTORIALS_REQUIRES,
        },
    )
