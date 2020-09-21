# Installing brainlit on iOS using conda environment:
1. Installation of conda environment
    * [get Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html)
    * create a virtual environment: `conda create --name brainlit python=3.8`
    * activate the environment: `conda activate brainlit`
2. Install brainlit
    * Option 1: install from pypi
        * install brainlit: `pip install brainlit`
    * Option 2: install brainlit from source
        * cd to directory you want to install brainlit in
        * clone the repo: git clone https://github.com/neurodata/brainlit.git
        * cd into the repo: `cd brainlit`
        * install brainlit: `pip install -e .`

# Downloading brains tutorial on iOS using conda environment, using jupyter notebook: 
* in a new terminal: `conda activate brainlit`
* cd to directory brainlit is installed
* `cd brainlit/docs/notebooks/utils`
* run jupyter notebook: `jupyter notebook`
* select `downloading_brains.ipynb`
* run tutorial

# Common issues and fixes:
* Issues with using a jupyter notebook
    * [fixes](https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html)
* Imports cause schema warning
    * ignore this
* Activation key issue
    * `cd ~/.cloudvolume/secrets`
    * Add a .txt file containing aws access key and secret access key (more detailed instructions under README.md)
    * Convert .txt file to .json 
* Section (1) Defining variables: Change directory names if they do not exist
    * Example 1:
        * Before: `dir = p + "mouse-light-viz/precomputed_volumes/brain1_2”`
        * After: `dir = p + "mouse-light-viz/precomputed_volumes/brain1"`
    * Example 2: 
        * Before: `dir_segments = p + "mouse-light-viz/precomputed_volumes/brain1_2_segments"`
        * After: `dir_segments = p + "mouse-light-viz/precomputed_volumes/brain1_segments"`
* Section (2) Create a Neuroglancer instance and download the volume: make sure variables are correct and functions have correct inputs
    * Example 3: 
        * Before: `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius, radius, radius)`
        * After:  `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius)`
* Section (4) View the volume: Kernel may crash, not allowing napari to be viewed
    * In terminal, type: `pip install opencv-contrib-python-headless`