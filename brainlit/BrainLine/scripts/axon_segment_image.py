
from cloudvolume import CloudVolume
from skimage import io
import numpy as np
import sys
import warnings
import subprocess
from tqdm import tqdm
import h5py
from joblib import Parallel, delayed
import multiprocessing
import os
import igneous.task_creation as tc
from taskqueue import LocalTaskQueue
from pathlib import Path
from brainlit.BrainLine.apply_ilastik import ApplyIlastik_LargeImage
from brainlit.BrainLine.data.axon_data import brain2paths

"""
Inputs
"""

brain = "test"
antibody_layer = "antibody"
background_layer = "background"
endogenous_layer = "endogenous"

threshold = 0.12  # threshold to use for ilastik
data_dir = str(Path.cwd().parents[0]) + "/brain_temp/" # data_dir = "/data/tathey1/matt_wright/brain_temp/"  # directory to store temporary subvolumes for segmentation

# Ilastik will run in "headless mode", and the following paths are needed to do so:
ilastik_path = "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh" # "/data/tathey1/matt_wright/ilastik/ilastik-1.4.0rc5-Linux/run_ilastik.sh"  # path to ilastik executable
ilastik_project = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/axon_segmentation.ilp" # "/data/tathey1/matt_wright/ilastik/model1/axon_segmentation.ilp"  # path to ilastik project


max_coords = [3072, 4352, 1792] #max coords or -1 if you want to process everything along that dimension
ncpu = 1 #16  # number of cores to use for detection
chunk_size = [256, 256, 256]#[256, 256, 300]


print(f"Number cpus available: {multiprocessing.cpu_count()}")
warnings.filterwarnings("ignore")
# """
# Segment Axon
# """
# layer_names = [antibody_layer, background_layer, endogenous_layer]
# alli = ApplyIlastik_LargeImage(ilastik_path = ilastik_path, ilastik_project=ilastik_project, ncpu=ncpu, object_type="axon")
# alli.apply_ilastik_parallel(brain_id=brain, layer_names=layer_names, threshold=threshold, data_dir=data_dir, chunk_size=chunk_size, max_coords=max_coords)
# alli.collect_axon_results(brain_id = brain, ng_layer_name="127.0.0.1:9010")


"""
Downsample Mask
"""
print("Downsampling...")
dir_base = brain2paths[brain]["base"]
layer_path = dir_base + "axon_mask"

tq = LocalTaskQueue(parallel=8)

tasks = tc.create_downsampling_tasks(
    layer_path,  # e.g. 'gs://bucket/dataset/layer'
    mip=0,  # Start downsampling from this mip level (writes to next level up)
    fill_missing=True,  # Ignore missing chunks and fill them with black
    axis="z",
    num_mips=5,  # number of downsamples to produce. Downloaded shape is chunk_size * 2^num_mip
    chunk_size=None,  # manually set chunk size of next scales, overrides preserve_chunk_size
    preserve_chunk_size=True,  # use existing chunk size, don't halve to get more downsamples
    sparse=False,  # for sparse segmentation, allow inflation of pixels against background
    bounds=None,  # mip 0 bounding box to downsample
    encoding=None,  # e.g. 'raw', 'compressed_segmentation', etc
    delete_black_uploads=True,  # issue a delete instead of uploading files containing all background
    background_color=0,  # Designates the background color
    compress="gzip",  # None, 'gzip', and 'br' (brotli) are options
    factor=(2, 2, 2),  # common options are (2,2,1) and (2,2,2)
)

tq.insert(tasks)
tq.execute()
