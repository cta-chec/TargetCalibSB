from setuptools import setup, find_packages

PACKAGENAME = "TargetCalibSB"
DESCRIPTION = (
    "Python package to explore calibration approaches for TARGET modules"
)
AUTHOR = "Jason J Watson"
AUTHOR_EMAIL = "jason.watson@physics.ox.ac.uk"
VERSION = "0.0.1"

setup(
    name=PACKAGENAME,
    packages=find_packages(),
    version=VERSION,
    description=DESCRIPTION,
    license='BSD3',
    install_requires=[
        'astropy',
        'scipy',
        'numpy',
        'matplotlib',
        'tqdm',
        'pandas>=0.21.0',
        'iminuit',
        'numba',
        'fitsio',
        'PyYAML',
        'packaging',
        'CHECLabPy',
    ],
    setup_requires=['pytest-runner', ],
    tests_require=['pytest', ],
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
)
