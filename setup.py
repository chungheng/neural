#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "Jinja2 >= 2.11",
    "matplotlib >= 3.3",
    "numpy >= 1.19",
    "scipy >= 1.5",
    "tqdm >= 4.48",
    "sympy >= 1.9",
    "numba >= 0.55",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Tingkai Liu",
    author_email="tingkai.liu@columbia.edu",
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    description="A simple python package for neuron and synapse models.",
    install_requires=requirements,
    extras_require={"cuda": ["cupy >= 10.1"]},
    license="BSD license",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="neural",
    name="neural",
    packages=find_packages(include=["neural", "neural.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/TK-21st/neural",
    version="0.1.0",
    zip_safe=False,
)
