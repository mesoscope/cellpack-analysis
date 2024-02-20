#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open("README.md") as readme_file:
    readme = readme_file.read()

setup_requirements = [
    "pytest-runner>=5.2",
]

test_requirements = [
    "black>=19.10b0",
    "codecov>=2.1.4",
    "flake8>=3.8.3",
    "flake8-debugger>=3.2.1",
    "pytest>=5.4.3",
    "pytest-cov>=2.9.0",
    "pytest-raises>=0.11",
]

dev_requirements = [
    *setup_requirements,
    *test_requirements,
    "bump2version>=1.0.1",
    "coverage>=5.1",
    "ipython>=7.15.0",
    "ipykernel",
    "m2r2>=0.2.7",
    "pytest-runner>=5.2",
    "Sphinx>=3.4.3",
    "sphinx_rtd_theme>=0.5.1",
    "tox>=3.15.2",
    "twine>=3.1.1",
    "wheel>=0.34.2",
]

requirements = [
    "aicscytoparam",
    "aicsimageio",
    # "cellpack",
    "cvapipe_analysis",
    "matplotlib",
    "numba>=0.57",
    "numpy",
    "pacmap",
    "pandas",
    "pathlib",
    "quilt3",
    "scipy",
    "scikit-image>=0.15.0",
    "scikit-learn",
    "seaborn",
    "tqdm",
    "trimesh",
    "umap-learn",
    "vtk==9.0.1",
]

extra_requirements = {
    "setup": setup_requirements,
    "test": test_requirements,
    "dev": dev_requirements,
    "all": [
        *requirements,
        *dev_requirements,
    ]
}

setup(
    author="Saurabh Mogre",
    author_email="saurabh.mogre@alleninstitute.org",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: Free for non-commercial use",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    description="Analysis of cellpack simulations",
    entry_points={
        "console_scripts": [
            "my_example=scripts.cellpack_PILR"
        ],
    },
    install_requires=requirements,
    license="Allen Institute Software License",
    long_description=readme,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords="cellpack_analysis",
    name="cellpack_analysis",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*"]),
    python_requires=">=3.8",
    setup_requires=setup_requirements,
    test_suite="aicscytoparam/tests",
    tests_require=test_requirements,
    extras_require=extra_requirements,
    # url="https://github.com/AllenCell/aics-cytoparam",
    # Do not edit this string manually, always use bumpversion
    # Details in CONTRIBUTING.rst
    version="0.0.1",
    zip_safe=False,
)
