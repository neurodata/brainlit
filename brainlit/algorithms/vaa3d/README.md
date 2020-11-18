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

- The input to Vaa3D will require a `.tif` file. `skimage.io` can be used to save `.tif` files.

- To run APP2 in Vaa3D, use the `runVaa3dPlugin` with the plugin name `Vaa3D_Neuron2` and function name `app2`

- The output of the Vaa3D run will be a `.swc` file in the same directory as the input image file

- Note that there will be 2 `.swc` file outputs, one with an `_ini` in the name and one with a coordinate, for example `_x82_y42_z29`. The latter is the proper label output, which will be sparse enough to match the manually labeled traces.

## Processing .swc file

The output of Vaa3D is a dense `.swc` file which labels many voxels that are candidates for foreground pixels. Note that this does not match the format of the ground truth data, which is a linear interpolation of manually selected points. This means that the neuron is represented by a curve in 3-Space rather than a cylindrical volume. I used `skimage.morphology.skeletonize_3d` to do this. 

- Note that when loading in the `.swc` file, the `x` and `z` dimensions will need to be swapped to correct a tranpose bug.

# Experiment Plan

### Data to be used

The open-neurodata S3 instance has 3 brains worth of data. I will see if I can make use of all 3 of these to perform tests. The first step will be to convert these image files into the requisite `.tif` format to run Vaa3D. 

EDIT: There is actually a benchmarking dataset with the proper ground truth labels. I will figure out how to get access to that and see if I can process the data into the proper formats necessary for running APP2.

### Algorithms to be compared

The primary goal is to compare APP2 to an existing ground-truth solution. I will also compare the results that come out of an algorithm in `adaptive_thresh.py` under the `generate_fragments` directory as a means of presenting a comparison to an existing algorithm. From preliminary testing, the Otsu segmentation algorithm has decent performance on the small demo dataset that I tried. It may be necessary to investigate the algorithm's behavior on a larger scale dataset to determine if it is usable.

### Further processing - Labels Skeletonization - NO LONGER NECESSARY

The labelsets that result from APP2 and Otsu Segmentation are known as dense labelsets. These annotate every voxel that the algorithms have determined are part of a neuron. The problem resides in this high density. The ground truth is a series of hand-labeled points, which is then linearly interpolated to form a rough skeleton of the neuron path. The outputs of APP2 and Otsu will need to be skeletonized to create a similar structure to that of the ground truth labels. This will ensure that the neuron traces have a similar density.

NOTE: It would be prudent to mimic the processes done in the APP2 paper, Peng et. al. 2013, as they presented skeletonized outputs. It appears that V3D from an [earlier paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857929/) has the ability to do skeletonization.

`skimage.morphology.skeletonize` provides a preliminary solution to this. The results are not as clean as I expected, but it may also be due to the low resolution of the demo dataset.

### Further processing - Labels resampling (already included in APP2 algorithm)

To prepare the labels for comparison metric,s a resampling process must be done so that individual labels are at least 1 micron or 1 voxel away from each other (depending on the density and modality) and at a fixed distance from each other. This will ensure a roughly uniform quality of trace across the neuron. However, APP2 already includes the resampling process. If resampling is needed, one can apply the resampling through the Vaa3D application under `/plugin/neuron_utilities/resample_swc`.

### Comparison metrics

Comparing `.swc` files will require a measure of distance.

- The first metric we can use is a notion of closeness to a given solution. For each point on an experimental skeleton, we can calculate its minimum distance to the closest point on the truth skeleton. This is effectively a 1-Nearest-Neighbor algorithm. The sum of all of these distances will be a "cost" for that skeleton.

- The [V3D paper](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2857929/) uses a concept known as spatial distance. The process is as follows:

  - Randomly sample nodes from neurons A and B
  - Calculate the euclidean distance from each sampled node in A to each sampled node in B, and average the,. This is known as the the directed divergence, DDIV(A,B)
  - To calculate the undirected spatial distance (SD) between A and B, we average DDIV(A,B) and DDIV(B,A)
  - Substantial Spatial Distance (SSD) further modifies this. It thresholds all of the distances first, keeping only those distances that are greater than 2 voxels. The percentage of "significant distances" is then calculated (kept/total).

- Robustness. If the above is implemented successfully, we can move forward and attempt to replicate the APP2 paper's signal deletion test, where voxels' intensities are randomly set to 0 in the source image. This will test the robustness of the algorithm to reproduce reconstructions in low overall intensity images. We can then run the algorithms with a 30%, 50%, 70% deletion and observe how the distance metrics hold up. I will do a further review of whether or not this type of test is necessary, as robustness tests are done to ensure the wider applicability of the tracing algorithm on future datasets. However, such a test might not be necessary for Brainlit specifically, as our data is expected to be of a certain quality and format since all of the Janelia brains are scanned using the same modality.

### Result Outputs

The resulting outputs will be tabulated in a similar way as [Peng Et. Al. 2013](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3661058/). After computing the SSD scores, we can place them on a bar graph to compare. The Y-axis will be the SSD score, and the X-axis will be the different testing datasets, with multiple color bars and a legend to denote to each algorithm used.

Deletion tests can be done in a similar way. However, it may be necessary to have an individual graph for each testing dataset for clarity. Y-axis will be SSD score, and X-axis will be deletion percentage. Multiple bar colors will be used alongside a legend to denote the algorithm used.

Visualization comparisons will also be useful, such as selected viewing windows near branches or the soma. The mouselight datasets have traditionally been human-graded, thus having a human grader who is familiar with neuron morphology assess algorithm correctness can allow for additional qualitative analysis.

### How do we expect the APP2 algorithm to perform?

APP2 has been used as a single-neuron tracing algorithm. As such, it may perform poorly with multi-neuron datasets. The algorithm involves the detection of a soma using a Gray-Weighted Image Distance Transform. The maximally transformed point is the center of the soma. However, this means that there will only exist 1 soma, as there is only 1 maximally transformed point. The APP2 paper does not detail what occurs if there are multiple neurons, or what happens to segments that are disconnected. I expect APP2 to trace a single neuron successfully, however I am are unsure of the behavior for the other neurons populating the space. It could be the case that we may need multiple passes of APP2 to remove the traced neuron by zeroing its intensity information, and extracting each neuron one by one.
