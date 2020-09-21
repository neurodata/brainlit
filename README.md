# Brainlit

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![Build Status](https://travis-ci.com/neurodata/brainlit.svg?branch=master)](https://travis-ci.com/neurodata/brainlit)
[![PyPI version](https://badge.fury.io/py/brainlit.svg)](https://badge.fury.io/py/brainlit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/neurodata/brainlit/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodata/brainlit)
![Docker Cloud Build Status](https://img.shields.io/docker/cloud/build/bvarjavand/brainlit)
![Docker Image Size (latest by date)](https://img.shields.io/docker/image-size/bvarjavand/brainlit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)  
This repository is a container of methods that Neurodata usees to expose their open-source code while it is in the process of being merged with larger scientific libraries such as scipy, scikit-image, or scikit-learn. Additionally, methods for computational neuroscience on brains too specific for a general scientific library can be found here, such as image registration software tuned specifically for large brain volumes.

![Brainlight Features](https://i.postimg.cc/QtG9Xs68/Brainlit.png)

- [Motivation](#motivation)
- [Installation](#installation)
  - [Environment](#environment)
  - [Install from pypi](#install-from-pypi)
  - [Install from source](#install-from-source)
- [How to Use Brainlit](#how-to-use-brainlit)
  - [Data Setup](#data-setup)
  - [Create a Session](#create-a-session)
- [Features](#features)
  - [Registration](#registration)
- [Core](#core)
  - [Push/Pull Data](#push-and-pull-data)
  - [Visualize](#visualize)
  - [Manually Segment](#manually-segment)
  - [Automatically Segment](#automatically-and-semi-automatically-segment)
- [API reference](#api-reference)
- [Tests](#tests)
- [Contributing](#contributing)
- [Credits](#credits)

## Motivation

The repository originated as the project of a team in Joshua Vogelstein's class **Neurodata** at Johns Hopkins University. This project was focused on data science towards the [mouselight data](https://www.hhmi.org/news/mouselight-project-maps-1000-neurons-and-counting-in-the-mouse-brain). It becme apparent that the tools developed for the class would be useful for other groups doing data science on large data volumes.
The repository can now be considered a "holding bay" for code developed by Neurodata for collaborators and researchers to use.

## Installation

### Operating Systems
Brainlit is compatible with iOS, Windows, and Unix systems.

#### Windows Linux Subsystem 2 
For Windows 10 users that prefer Linux functionality without the speed sacrifice of a VM, Brainlit can be installed and run on WSL2. See installation walkthrough [here.](WSL2-install-instructions.md)

### Environment

#### (optional, any python >= 3.8 environment will suffice)

- [get conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
- create a virtual environment: `conda create --name brainlit python=3.8`
- activate the environment: `conda activate brainlit`

### Install from pypi

- install brainlit: `pip install brainlit`

### Install from source

- clone the repo: `git clone https://github.com/neurodata/brainlit.git`
- cd into the repo: `cd brainlit`
- install brainlit: `pip install -e .`

## How to use Brainlit

### Data setup

The `source` data directory should have an octree data structure

```
 data/
├── default.0.tif
├── transform.txt
├── 1/
│   ├── 1/, ..., 8/
│   └── default.0.tif
├── 2/ ... 8/
└── consensus-swcs (optional)
    ├── G-001.swc
    ├── G-002.swc
    └── default.0.tif
```

If your team wants to interact with cloud data, each member will need account credentials specified in `~/.cloudvolume/secrets/x-secret.json`, where `x` is one of `[aws, gc, azure]` which contains your id and secret key for your cloud platform.
We provide a template for `aws` in the repo for convenience.

### Create a session

Each user will start their scripts with approximately the same lines:

```
from brainlit.utils.ngl import NeuroglancerSession

session = NeuroglancerSession(url='file:///abc123xyz')
```

From here, any number of tools can be run such as the visualization or annotation tools. [Interactive demo](https://github.com/neurodata/brainlit/blob/master/docs/notebooks/visualization/visualization.ipynb).

## Features

### Registration

The registration subpackage is a facsimile of ARDENT, a pip-installable (pip install ardent) package for nonlinear image registration wrapped in an object-oriented framework for ease of use. This is an implementation of the LDDMM algorithm with modifications, written by Devin Crowley and based on "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." This paper extends on an older LDDMM paper, "Computing large deformation metric mappings via geodesic flows of diffeomorphisms."

This is the more recent paper:

Tward, Daniel, et al. "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." Frontiers in neuroscience 14 (2020).

https://doi.org/10.3389/fnins.2020.00052

This is the original LDDMM paper:

Beg, M. Faisal, et al. "Computing large deformation metric mappings via geodesic flows of diffeomorphisms." International journal of computer vision 61.2 (2005): 139-157.

https://doi.org/10.1023/B:VISI.0000043755.93987.aa

A tutorial is available in docs/notebooks/registration_demo.ipynb.

## Core

The core brain-lit package can be described by the diagram at the top of the readme:

### (Push and Pull Data)

Brainlit uses the Seung Lab's [Cloudvolume](https://github.com/seung-lab/cloud-volume) package to push and pull data through the cloud or a local machine in an efficient and parallelized fashion. [Interactive demo](https://github.com/neurodata/brainlit/blob/master/docs/notebooks/utils/uploading_brains.ipynb).  
The only requirement is to have an account on a cloud service on s3, azure, or google cloud.

Loading data via local filepath of an octree structure is also supported. [Interactive demo](https://github.com/neurodata/brainlit/blob/master/docs/notebooks/utils/upload_brains.ipynb).

### Visualize

Brainlit supports many methods to visualize large data. Visualizing the entire data can be done via Google's [Neuroglancer](https://github.com/google/neuroglancer), which provides a web link as shown below.

screenshot

Brainlit also has tools to visualize chunks of data as 2d slices or as a 3d model. [Interactive demo](https://github.com/neurodata/brainlit/blob/master/docs/notebooks/visualization/visualization.ipynb).

screenshot

### Manually Segment

Brainlit includes a lightweight manual segmentation pipeline. This allows collaborators of a projec to pull data from the cloud, create annotations, and push their annotations back up as a separate channel. [Interactive demo](https://github.com/neurodata/brainlit/blob/master/docs/notebooks/pipelines/manual_segementation.ipynb).

### Automatically and Semi-automatically Segment

Similar to the above pipeline, segmentations can be automatically or semi-automatically generated and pushed to a separate channel for viewing. [Interactive demo](https://github.com/neurodata/brainlit/blob/master/docs/notebooks/pipelines/seg_pipeline_demo.ipynb).

## API Reference

[![Documentation Status](https://readthedocs.org/projects/brainlight/badge/?version=latest)](https://brainlight.readthedocs.io/en/latest/?badge=latest)
The documentation can be found at [https://brainlight.readthedocs.io/en/latest/](https://brainlight.readthedocs.io/en/latest/).

## Tests

Running tests can easily be done by moving to the root directory of the brainlit package ant typing `pytest tests` or `python -m pytest tests`.  
Running a specific test, such as `test_upload.py` can be done simply by `ptest tests/test_upload.py`.

## Contributing

Contribution guidelines can be found via [CONTRIBUTING.md](https://github.com/neurodata/brainlit/blob/master/CONTRIBUTING.md)

## Credits

Thanks to the neurodata team and the group in the neurodata class which started the project.
This project is currently managed by Tommy Athey and Bijan Varjavand.
