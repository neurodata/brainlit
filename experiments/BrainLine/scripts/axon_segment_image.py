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

"""
Inputs
"""

brain = "MS9"
antibody_layer = "Ch_647"
background_layer = "Ch_561"
endogenous_layer = "Ch_488"

threshold = 0.22  # threshold to use for ilastik
brainline_exp_dir = Path(os.getcwd()) / Path(__file__).parents[1]
data_dir = (
    brainline_exp_dir / "data" / "brain_temp"
)  # data_dir = "/data/tathey1/matt_wright/brain_temp/"  # directory to store temporary subvolumes for segmentation
data_file = brainline_exp_dir / "data" / "axon_data.json"

# Ilastik will run in "headless mode", and the following paths are needed to do so:
ilastik_path = "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh"  # "/data/tathey1/matt_wright/ilastik/ilastik-1.4.0rc5-Linux/run_ilastik.sh"  # path to ilastik executable
ilastik_project = brainline_exp_dir / "data" / "models" / "axon" / "axon_segmentation.ilp"  # "/data/tathey1/matt_wright/ilastik/model1/axon_segmentation.ilp"  # path to ilastik
ilastik_path = "/home/user/Documents/ilastik-1.4.0-Linux/run_ilastik.sh"



min_coords = [
    836,
    -1,
    -1,
]  # max coords or -1 if you want to process everything along that dimension
max_coords = [
    6635,
    -1,
    -1,
]  # max coords or -1 if you want to process everything along that dimension
ncpu = 2  # number of cores to use for detection
chunk_size = [512, 1024, 2048]  # [256, 256, 300]


print(f"Number cpus available: {multiprocessing.cpu_count()}")
warnings.filterwarnings("ignore")
"""
Segment Axon
"""
layer_names = [antibody_layer, background_layer, endogenous_layer]
alli = ApplyIlastik_LargeImage(
    ilastik_path=ilastik_path,
    ilastik_project=ilastik_project,
    ncpu=ncpu,
    data_file=data_file,
)
alli.apply_ilastik_parallel(
    brain_id=brain,
    layer_names=layer_names,
    threshold=threshold,
    data_dir=data_dir,
    chunk_size=chunk_size,
    min_coords=min_coords,
    max_coords=max_coords,
)
alli.collect_axon_results(brain_id=brain, ng_layer_name="Ch_647")


"""
Downsample Mask
"""
# Cloudreg only works on complete volumes, so you cannot delete_black_uploads to allow missing data etc.
downsample_ask = input(
    f"Do you want to downsample the axon_mask for brain {brain}? (y/n)"
)
if downsample_ask == "y":
    print("Downsampling...")
    with open(data_file) as f:
        js = json.open(f)
        dir_base = js["brain2paths"][brain]["base"]

    layer_path = dir_base + "axon_mask"

    tq = LocalTaskQueue(parallel=16)

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
        delete_black_uploads=False,  # issue a delete instead of uploading files containing all background
        background_color=0,  # Designates the background color
        compress="gzip",  # None, 'gzip', and 'br' (brotli) are options
        factor=(2, 2, 2),  # common options are (2,2,1) and (2,2,2)
    )

    tq.insert(tasks)
    tq.execute()

"""
Making info files for transformed images
"""
# make_trans_layers = input(
#     f"Will you be transforming axon_mask into atlas space? (should relevant info files be made) (y/n)"
# )

# if make_trans_layers == "y":
#     atlas_vol = CloudVolume(
#         "precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017"
#     )
#     layer_path = brain2paths[brain]["base"] + "axon_mask_transformed"
#     print(f"Writing info file at {layer_path}")
#     info = CloudVolume.create_new_info(
#         num_channels=1,
#         layer_type="image",
#         data_type="uint16",  # Channel images might be 'uint8'
#         encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
#         resolution=atlas_vol.resolution,  # Voxel scaling, units are in nanometers
#         voxel_offset=atlas_vol.voxel_offset,
#         chunk_size=[32, 32, 32],  # units are voxels
#         volume_size=atlas_vol.volume_size,  # e.g. a cubic millimeter dataset
#     )
#     vol_mask = CloudVolume(layer_path, info=info)
#     vol_mask.commit_info()
