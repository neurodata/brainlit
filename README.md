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

- [Brainlit](#brainlit)
  - [Motivation](#motivation)
  - [Installation](#installation)
    - [Environment](#environment)
      - [(optional, any python >= 3.8 environment will suffice)](#optional-any-python--38-environment-will-suffice)
    - [Install from pypi](#install-from-pypi)
    - [Install from source](#install-from-source)
  - [How to use Brainlit](#how-to-use-brainlit)
    - [Data setup](#data-setup)
    - [Create a session](#create-a-session)
  - [Features](#features)
    - [Registration](#registration)
  - [Core](#core)
    - [(Push and Pull Data)](#push-and-pull-data)
    - [Visualize](#visualize)
    - [Manually Segment](#manually-segment)
    - [Automatically and Semi-automatically Segment](#automatically-and-semi-automatically-segment)
  - [API Reference](#api-reference)
  - [Tests](#tests)
  - [Common errors and troubleshooting](#common-errors-and-troubleshooting)
    - [AWS credentials](#aws-credentials)
      - [Missing `AWS_ACCESS_KEY_ID`](#missing-aws_access_key_id)
      - [Empty `AKID` (Access Key ID)](#empty-akid-access-key-id)
      - [Access denied](#access-denied)
  - [Contributing](#contributing)
  - [Credits](#credits)

## Motivation

The repository originated as the project of a team in Joshua Vogelstein's class **Neurodata** at Johns Hopkins University. This project was focused on data science towards the [mouselight data](https://www.hhmi.org/news/mouselight-project-maps-1000-neurons-and-counting-in-the-mouse-brain). It becme apparent that the tools developed for the class would be useful for other groups doing data science on large data volumes.
The repository can now be considered a "holding bay" for code developed by Neurodata for collaborators and researchers to use.

## Installation

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

## Common errors and troubleshooting

- [iOS Install/Run Issues](https://github.com/NeuroDataDesign/brainlit/blob/aws-keys-docs-jacopo/docs/iOS_Install_%26_Run_Issues.md)

### AWS credentials

:warning: **SECURITY DISCLAIMER** :warning:
Do **NOT** push any official AWS credentials to any repository. These posts are a good reference to get a sense of what pushing AWS credentials implies:

1. *I Published My AWS Secret Key to GitHub* by Danny Guo [https://www.dannyguo.com/blog/i-published-my-aws-secret-key-to-github/](https://www.dannyguo.com/blog/i-published-my-aws-secret-key-to-github/)
2. *Exposing your AWS access keys on Github can be extremely costly. A personal experience.* by Guru [https://medium.com/@nagguru/exposing-your-aws-access-keys-on-github-can-be-extremely-costly-a-personal-experience-960be7aad039](https://medium.com/@nagguru/exposing-your-aws-access-keys-on-github-can-be-extremely-costly-a-personal-experience-960be7aad039)
3. *Dev put AWS keys on Github. Then BAD THINGS happened* by Darren Pauli [https://www.theregister.com/2015/01/06/dev_blunder_shows_github_crawling_with_keyslurping_bots/](https://www.theregister.com/2015/01/06/dev_blunder_shows_github_crawling_with_keyslurping_bots/)


Brainlit can access data volumes stored in [AWS S3](https://aws.amazon.com/free/storage/s3/?trk=ps_a134p000006BgagAAC&trkCampaign=acq_paid_search_brand&sc_channel=ps&sc_campaign=acquisition_US&sc_publisher=google&sc_category=storage&sc_country=US&sc_geo=NAMER&sc_outcome=acq&sc_detail=aws%20s3&sc_content=S3_e&sc_segment=432339156183&sc_medium=ACQ-P|PS-GO|Brand|Desktop|SU|Storage|Product|US|EN|Text&s_kwcid=AL!4422!3!432339156183!e!!g!!aws%20s3&ef_id=CjwKCAjwkoz7BRBPEiwAeKw3q7yLVNTPLORSa7QUsB5aGT0wAKrnrlnkwNPex8vdqYMVBPqgjlZV2RoCIdgQAvD_BwE:G:s&s_kwcid=AL!4422!3!432339156183!e!!g!!aws%20s3) through the [CloudVolume](https://github.com/seung-lab/cloud-volume) package. As specified in the [docs](https://github.com/seung-lab/cloud-volume#credentials), AWS credentials have to be stored in a file called `aws-secret.json` inside the `~.cloudvolume/secrets/` folder.

Prerequisites to successfully troubleshoot errors related to AWS credentials:

- [ ] The data volume is hosted on S3 (i.e. the link looks like `s3://your-bucket-name/some-path/some-folder`).
- [ ] Familiarity with [IAM Roles](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles.html) and [how to create them](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_create.html).
- [ ] An `AWS_ACCESS_KEY_ID` and an `AWS_SECRET_ACCESS_KEY` with adequate permissions, provided by the system administrator. Brainlit does not require the IAM user associated with the credentials to have access to the AWS console (i.e. it can be a service account).

Here is a collection of known issues, along with their troubleshoot guide:

#### Missing `AWS_ACCESS_KEY_ID`

Error message:

```python
~/opt/miniconda3/envs/brainlit/lib/python3.8/site-packages/cloudvolume/connectionpools.py in _create_connection(self)
     99       return boto3.client(
    100         's3',
--> 101         aws_access_key_id=self.credentials['AWS_ACCESS_KEY_ID'],
    102         aws_secret_access_key=self.credentials['AWS_SECRET_ACCESS_KEY'],
    103         region_name='us-east-1',

KeyError: 'AWS_ACCESS_KEY_ID'
```

This error is thrown when the `credentials` object has an empty `AWS_ACCESS_KEY_ID` entry. This probably indicates that `aws-secret.json`  is not stored in the right folder and it cannot be found by CloudVolume. Make sure your credential file is named correctly and stored in `~.cloudvolume/secrets/`. If you are a Windows user, the output of this Python snippet is the expansion of `~` for your system:

```python
import os
HOME = os.path.expanduser('~')
print(HOME)
```

example output:

```bash
Python 3.8.3 (v3.8.3:6f8c8320e9)
>>> import os
>>> HOME = os.path.expanduser('~')
>>> print(HOME)
C:\Users\user
```

#### Empty `AKID` (Access Key ID)

Error message:

```python
/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
    654             error_code = parsed_response.get("Error", {}).get("Code")
    655             error_class = self.exceptions.from_code(error_code)
--> 656             raise error_class(parsed_response, operation_name)
    657         else:
    658             return parsed_response
ClientError: An error occurred (AuthorizationHeaderMalformed) when calling the GetObject operation: The authorization header is malformed; a non-empty Access Key (AKID) must be provided in the credential.
```

This error is thrown when your `aws-secret.json` file is stored and loaded correctly, and it looks like this:

```json
{
  "AWS_ACCESS_KEY_ID": "",
  "AWS_SECRET_ACCESS_KEY": ""
}
```

Even though the bucket itself may be public, [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) requires some non-empty AWS credentials to instantiante the S3 API client.

#### Access denied

```python
/Library/Frameworks/Python.framework/Versions/3.8/lib/python3.8/site-packages/botocore/client.py in _make_api_call(self, operation_name, api_params)
    654             error_code = parsed_response.get("Error", {}).get("Code")
    655             error_class = self.exceptions.from_code(error_code)
--> 656             raise error_class(parsed_response, operation_name)
    657         else:
    658             return parsed_response
ClientError: An error occurred (AccessDenied) when calling the GetObject operation: Access Denied
```

This error is thrown when:

1. The AWS credentials are stored and loaded correctly but are not allowed to access the data volume. A check with the system administrator is required.

2. There is a typo in your credentials. The content of `aws-secret.json` should look like this:

```json
{
  "AWS_ACCESS_KEY_ID": "$YOUR_AWS_ACCESS_KEY_ID",
  "AWS_SECRET_ACCESS_KEY": "$AWS_SECRET_ACCESS_KEY"
}
```

where the `$` are placeholder characters and should be replaced along with the rest of the string with the official AWS credentials.

## Contributing

Contribution guidelines can be found via [CONTRIBUTING.md](https://github.com/neurodata/brainlit/blob/master/CONTRIBUTING.md)

## Credits

Thanks to the neurodata team and the group in the neurodata class which started the project.
This project is currently managed by Tommy Athey and Bijan Varjavand.
