from cloudvolume import CloudVolume, exceptions
from skimage import io, measure
import numpy as np
import sys
import warnings
import subprocess
from tqdm import tqdm
import h5py
from joblib import Parallel, delayed
import multiprocessing
import os
from brainlit.BrainLine.data.soma_data import brain2paths
from brainlit.BrainLine.apply_ilastik import ApplyIlastik_LargeImage
from pathlib import Path

""" 
Inputs
"""
# DOUBLE CHECK:
# -dir_base
# data_dir and results_dir ARE CLEAR
# threshold IS CORRECT
brain = "test"
antibody_layer = "antibody"
background_layer = "background"
endogenous_layer = "endogenous"

threshold = 0.28  # threshold to use for ilastik
data_dir = str(Path.cwd().parents[0]) + "/brainr_temp/" # "/data/tathey1/matt_wright/brainr_temp/"  # directory to store temporary subvolumes for segmentation
results_dir = str(Path.cwd().parents[0]) + "/brainr_results/"  # directory to store coordinates of soma detections

# Ilastik will run in "headless mode", and the following paths are needed to do so:
ilastik_path = "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh" #"/data/tathey1/matt_wright/ilastik/ilastik-1.4.0rc5-Linux/run_ilastik.sh"  # path to ilastik executable
ilastik_project = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/matt_soma_rabies_pix_3ch.ilp" #"/data/tathey1/matt_wright/ilastik/soma_model/matt_soma_rabies_pix_3ch.ilp"  # path to ilastik project

max_coords = [3072, 4352, 1792] #max coords or -1 if you want to process everything along that dimension
ncpu = 10 #16  # number of cores to use for detection
chunk_size = [256, 256, 256]#[256, 256, 300]

print(f"Number cpus available: {multiprocessing.cpu_count()}")
warnings.filterwarnings("ignore")

""" 
Detect Somas
"""

layer_names = [antibody_layer, background_layer, endogenous_layer]
alli = ApplyIlastik_LargeImage(ilastik_path = ilastik_path, ilastik_project=ilastik_project, ncpu=ncpu)
alli.apply_ilastik_parallel(brain_id=brain, layer_names=layer_names, threshold=threshold, data_dir=data_dir, results_dir=results_dir, chunk_size=chunk_size, max_coords=max_coords)
alli.collect_results(brain_id=brain)
