import os
import sys
from setuptools import setup, find_packages
from sys import platform

PACKAGE_NAME = "brainlit"
DESCRIPTION = "Code to process and analyze brainlit data"
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = (
    "Thomas Athey, Bijan Varjivand, Ryan Lu, Matt Figdore, Alex Fiallos, Stanley Wang, Victor Wang",
)
AUTHOR_EMAIL = "tathey1@jhu.edu"
URL = "https://github.com/neurodata/brainlit"
MINIMUM_PYTHON_VERSION = 3, 7  # Minimum of Python 3.7
REQUIRED_PACKAGES = [
    "CloudReg",
    "aicspylibczi>=3.0.5",
    "brainrender",
    "numpy>=1.8.1",
    "scikit-image>=0.16.2",
    "networkx>=2.1",
    "scikit-learn==1.1.2",  # issues with pairwise_distances_argmin_min (e.g. used in NeuronTrace) on other versions
    "scipy>=1.1.0",
    "seaborn>=0.9.0",
    "tifffile>=2020.7.17",
    "napari[pyqt5]>=0.2.11",
    "PyQt5<=5.15.7",  # bc there was an error with 5.15.8, can remove once that's resolved
    "cloud-volume>=4.2.0",
    "feather-format==0.4.1",
    "ome-zarr>=0.6.0",
    "zarr>=2.10.2",
    "h5py>=3.3.0",
    # "pcurvepy @ git+https://git@github.com/CaseyWeiner/pcurvepy@master#egg=pcurvepy",
    "similaritymeasures>=0.4.4",
    "statannotations>=0.4.4",
]

# Find savanna version.
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
for line in open(os.path.join(PROJECT_PATH, "brainlit", "__init__.py")):
    if line.startswith("__version__ = "):
        VERSION = line.strip().split()[2][1:-1]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license="Apache License 2.0",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        "napari.manifest": [
            "brainlit = brainlit.napari_viterbrain:napari.yaml",
        ],
    },
    package_data={"brainlit.napari_viterbrain": ["napari.yaml"]},
)
