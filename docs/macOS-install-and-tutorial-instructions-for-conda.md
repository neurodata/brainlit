# Installing brainlit on macOS using conda environment:
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

# Downloading brains tutorial on macOS using conda environment, using jupyter notebook: 
* in a new terminal: `conda activate brainlit`
* cd to directory brainlit is installed
* `cd brainlit/docs/notebooks/utils`
* run jupyter notebook: `jupyter notebook`
* select `downloading_brains.ipynb`
* run tutorial

# Common issues and fixes:
* Issues with using a jupyter notebook
    * [fixes](https://jupyter-notebook.readthedocs.io/en/stable/troubleshooting.html)
* Imports section: cause schema warning
    * This warning doesn't cause any errors, and the code can still run unaffected.
* AWS Credendials Issues
    * Refer to the instructions on: https://github.com/NeuroDataDesign/brainlit/blob/develop/docs/AWS_Credentials_Issues.md
* Section (2) Create a Neuroglancer instance and download the volume: make sure variables are correct and functions have correct inputs
    * Example 3: 
        * Before: `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius, radius, radius)`
        * After:  `img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius)`
* Section (4) View the volume: Kernel may crash, not allowing napari to be viewed
    * In terminal, type: `pip install opencv-contrib-python-headless`