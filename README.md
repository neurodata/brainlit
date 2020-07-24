# Brainlit
Link to mouselight article [here](https://www.hhmi.org/news/mouselight-project-maps-1000-neurons-and-counting-in-the-mouse-brain).

[![Python](https://img.shields.io/badge/python-3.7-blue.svg)]()
[![Build Status](https://travis-ci.com/neurodata/brainlit.svg?branch=master)](https://travis-ci.com/neurodata/brainlit)
[![Documentation Status](https://readthedocs.org/projects/brainlight/badge/?version=latest)](https://brainlight.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/neurodata/brainlit/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodata/brainlit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

# Overview
Tools to read and analyze brainlit data sets.

## Brainlight Features
![Brainlight Features](https://raw.githubusercontent.com/neurodata/brainlight/diagram/Brainlight.png)

### Registration
The registration subpackage is a facsimile of ARDENT, a pip-installable (pip install ardent) package for nonlinear image registration wrapped in an object-oriented framework for ease of use. This is an implementation of the LDDMM algorithm with modifications, written by Devin Crowley and based on "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." This paper extends on an older LDDMM paper, "Computing large deformation metric mappings via geodesic flows of diffeomorphisms."

This is the more recent paper:

Tward, Daniel, et al. "Diffeomorphic registration with intensity transformation and missing data: Application to 3D digital pathology of Alzheimer's disease." Frontiers in neuroscience 14 (2020).

https://doi.org/10.3389/fnins.2020.00052

This is the original LDDMM paper:

Beg, M. Faisal, et al. "Computing large deformation metric mappings via geodesic flows of diffeomorphisms." International journal of computer vision 61.2 (2005): 139-157.

https://doi.org/10.1023/B:VISI.0000043755.93987.aa

A tutorial is available in docs/notebooks/registration_demo.ipynb.
