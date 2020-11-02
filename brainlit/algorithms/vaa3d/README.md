# WORK IN PROGRESS

## Installation process:

- Download the Windows64 release version of [Vaa3D v3.447](https://github.com/Vaa3D/release/releases/tag/v3.447)
- Clone the [pyVaa3D](https://github.com/ajkswamy/pyVaa3d) plugin repository
- Activate the Brainlit `conda` environment, `cd` into the pyVaa3D directory, and `pip install -e .`
- The original instructions can be found [here](https://github.com/ajkswamy/pyVaa3d/blob/master/INSTALL.md).

## Initial Run

- Import the method `from pyVaa3d.vaa3dWrapper import runVaa3dPlugin`

- When importing this function, the terminal will prompt the user to enter the path to the executable `start_vaa3d.sh`. Note that on Windows, the file is actually `vaa3d_msvc.exe`, located inside the `C:Users\...\Vaa3D_V3.447_Windows_MSVC_64bit\` directory. This only needs to be done once, and is cached. 

- If it loads correctly, the Python terminal should look something like this:
[![vaa3d-executable-located.png](https://i.postimg.cc/LsrYK2wD/vaa3d-executable-located.png)](https://postimg.cc/mcy27fb1)
  Note that the highlighted portion is a user input.

## Running APP2

- See python script (will update later on some other notes)

- The current version has a problem where the labels are flipped along y=x, will need to fix this.

# Experiment Plan

### Data to be used

The open-neurodata S3 instance has 3 brains worth of data. I will see if I can make use of all 3 of these to perform tests.

### Algorithms to be compared

The primary goal is to compare APP2 to an existing ground-truth solution. I will also compare the results that come out of an algorithm in `adaptive_thresh.py` under the `generate_fragments` directory as a means of presenting a comparison to an existing algorithm. From preliminary testing, the Otsu segmentation algorithm has decent performance on the small demo dataset that I tried. It may be necessary to investigate the algorithm's behavior on a larger scale dataset to determine if it is usable.

### Further processing

The labelsets that result from APP2 and Otsu Segmentation are known as dense labelsets. These annotate every voxel that the algorithms have determined are part of a neuron. The problem resides in this high density. The ground truth is a series of hand-labeled points, which is then linearly interpolated to form a rough skeleton of the neuron path. The outputs of APP2 and Otsu will need to be skeletonized to create a similar structure to that of the ground truth labels. This will ensure that the neuron traces have a similar density.

NOTE: It would be prudent to mimic the processes done in the APP2 paper, Peng et. al. 2013, as they presented skeletonized outputs. It appears that V3D from an [earlier paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857929/) has the ability to do skeletonization.

### Comparison metrics

Comparing `.swc` files will require a measure of distance.

- The first metric we can use is a notion of closeness to a given solution. For each point on an experimental skeleton, we can calculate its minimum distance to the closest point on the truth skeleton. This is effectively a 1-Nearest-Neighbor algorithm. The sum of all of these distances will be a "cost" for that skeleton.

- The [V3D paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857929/) uses a concept known as spatial distance. The process is as follows:

  - Randomly sample nodes from neurons A and B
  - Calculate the euclidean distance from each sampled node in A to each sampled node in B, and average the,. This is known as the the directed divergence, DDIV(A,B)
  - To calculate the undirected spatial distance (SD) between A and B, we average DDIV(A,B) and DDIV(B,A)
  - Substantial Spatial Distance (SSD) further modifies this. It thresholds all of the distances first, keeping only those distances that are greater than 2 voxels. The percentage of "significant distances" is then calculated (kept/total).

- Robustness. If all goes well, we can attempt to replicate the APP2 paper's signal deletion test, where voxels' intensities are randomly set to 0 in the source image. This will test the robustness of the algorithm to reproduce reconstructions in low overall intensity images. We can then run the algorithms with a 30%, 50%, 70% deletion and observe how the distance metrics hold up.
