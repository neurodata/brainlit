# Brainlit
This repository is a container of methods that Neurodata usees to expose their open-source code while it is in the process of being merged with larger scientific libraries such as scipy, scikit-image, or scikit-learn. Additioanlly, methods for computational neuroscience on brains too specific for a general scientific library can be found here, such as image registration software tuned specifically for large brain volumes.

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![Build Status](https://travis-ci.com/neurodata/brainlit.svg?branch=master)](https://travis-ci.com/neurodata/brainlit)
[![codecov](https://codecov.io/gh/neurodata/brainlit/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodata/brainlit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

- [Motivation](#motivation)
- [Build Status](#build-status)
- [Code Style](#code-style)
- [Features](#features)
  * [Push/Pull Data](#push-and-pull-data)
  * [Visualize](#visualize)
  * [Manually Segment](#manually-segment)
  * [Automatically Segment](#automatically-and-semi-automatically-segment)
- [Installation](#installation)
- [API reference](#api-reference)
- [Tests](#tests)
- [How to Use Brainlit](#how-to-use-brainlit)
  * [Data Setup](#data-setup)
  * [Create a Session](#create-a-session)
- [Contribute](#contribute)
- [Credits](#credits)
- [License](#license)


## Motivation
The repository originated as the project of a team in Joshua Vogelstein's clsss **Neurodata** at Johns Hopkins University. This project was focused on data science towards the [mouselight data](https://www.hhmi.org/news/mouselight-project-maps-1000-neurons-and-counting-in-the-mouse-brain). It becme aparrent that the tools developed for the class would be useful for other groups doing data science on large data volumes.
The repository can now be considered a "holding bay" for code developed by Neruodata for collaborators and researchers to use.

## Build Status
[![Build Status](https://travis-ci.com/neurodata/brainlit.svg?branch=master)](https://travis-ci.com/neurodata/brainlit)  

## Code Style
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)  
Code in this project is formatted via the `black` auto-formatter.

## Features
The core brain-lit package can be described by the diagram below:

![Brainlight Features](https://raw.githubusercontent.com/neurodata/brainlight/diagram/Brainlight.png)

### (Push and Pull Data)
Brainlit uses the Seung Lab's ![Cloudvolume](https://github.com/seung-lab/cloud-volume) package to push and pull data through the cloud or a local machine in an efficient and parallelized fashion. ![Interactive demo]().  
The only requirement is to have an account on a cloud service on s3, azure, or google cloud.

Loading data via local filepath of an octree structure is also supported. ![Interactive demo]().

### Visualize
Brainlit supports many methods to visualize large data. Visualizing the entire data can be done via Google's ![Neuroglancer](https://github.com/google/neuroglancer), which provides a web link as shown below.

screenshot

Brainlit also has tools to visualize chunks of data as 2d slices or as a 3d model. ![Interactive demo]().

screenshot

### Manually Segment
Brainlit includes a lightweight manual segmentation pipeline. This allows collaborators of a projec to pull data from the cloud, create annotations, and push their annotations back up as a separate channel. ![Interactive demo]().

### Automatically and Semi-automatically Segment
Similar to the above pipeline, segmentations can be automatically or semi-automatically generated and pushed to a separate channel for viewing. ![Interactive demo](). 

## Installation
(pypi release badge)
Simply create and activate a python3.7 or greater virtual environment via conda or virtualenv, and either `pip install brainlit`, or
```
git clone
cd brainlit
pip install -e .
```

## API Reference
[![Documentation Status](https://readthedocs.org/projects/brainlight/badge/?version=latest)](https://brainlight.readthedocs.io/en/latest/?badge=latest)
The documentation can be found at ![https://brainlight.readthedocs.io/en/latest/](https://brainlight.readthedocs.io/en/latest/).

## Tests
Running tests can easily be done by moving to the root directory of the brainlit package ant typing `pytest tests` or `python -m pytest tests`.  
Running a specific test, such as `test_upload.py` can be done simply by `ptest tests/test_upload.py`.

## How to use Brainlit
### Data setup
First, decide for your team how you'd like to store the data - whether it will be on a local machine or on the cloud. If on the cloud,
each collaborator will need to create a file at `~/.cloudvolume/secrets/x-secret.json`, where `x` is one of `[aws, gc, azure]` which contains your id and secret key for your cloud platform.
### Create a session
Each user will start their scripts with approximately the same lines:
```
from brainlit.utils.ngl import NeuroglancerSession

session = NeuroglancerSession(url='file:///abc123xyz')
```
From here, any number of tools can be run such as the visualization or annotation tools.

## Contribute
TODO create contributing.md

## Credits
Thanks to the neurodata team and the group in the neurodata class which started the project.
This project is currently managed by Tommy Athey and Bijan Varjavand.

## License
Apache © Neurodata
