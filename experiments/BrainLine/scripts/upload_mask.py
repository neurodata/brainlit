from cloudvolume import CloudVolume
from brainlit.BrainLine.apply_ilastik import ApplyIlastik_LargeImage
from brainlit.BrainLine.util import _get_corners
import json
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm

local_vol = CloudVolume("precomputed://file:///mnt/data/Neuroglancer_Data/2023_04_10/MS12/axon_mask")

brain_id = "MS12"
antibody_layer = "Ch_647"
background_layer = "Ch_561"
endogenous_layer = "Ch_488"

brainline_exp_dir = Path(os.getcwd()) / Path(__file__).parents[1]
data_dir = (
    brainline_exp_dir / "data" / "brain_temp"
)  # data_dir = "/data/tathey1/matt_wright/brain_temp/"  # directory to store temporary subvolumes for segmentation
data_file = brainline_exp_dir / "data" / "axon_data.json"
layer_names = [antibody_layer, background_layer, endogenous_layer]
chunk_size = [1024, 1024, 1024]  # [256, 256, 300]
ncpu = 2


ilastik_path = "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh"  # "/data/tathey1/matt_wright/ilastik/ilastik-1.4.0rc5-Linux/run_ilastik.sh"  # path to ilastik executable
ilastik_project = brainline_exp_dir / "data" / "models" / "axon" / "axon_segmentation.ilp"  # "/data/tathey1/matt_wright/ilastik/model1/axon_segmentation.ilp"  # path to ilastik
ilastik_path = "/home/user/Documents/ilastik-1.4.0-Linux/run_ilastik.sh"

alli = ApplyIlastik_LargeImage(
    ilastik_path=ilastik_path,
    ilastik_project=ilastik_project,
    ncpu=ncpu,
    data_file=data_file,
)



with open(data_file) as f:
    data = json.load(f)
object_type = data["object_type"]
brain2paths = data["brain2paths"]

volume_base_dir_write = brain2paths[brain_id]["base_s3"]
volume_base_dir_read = brain2paths[brain_id]["base_local"]
mask_dir = volume_base_dir_write + "axon_mask"

sample_path = volume_base_dir_read + layer_names[1]
vol = CloudVolume(sample_path, parallel=True, mip=0, fill_missing=True)


try:
    CloudVolume(mask_dir)
except:
    assert np.all([c_ilastik % c_vol == 0 for c_ilastik, c_vol in zip(chunk_size, [128, 128, 2])])
    alli._make_mask_info(mask_dir, vol, [128, 128, 2])

s3_vol = CloudVolume(mask_dir)

corners = _get_corners(local_vol.shape, local_vol.chunks)

for corner in tqdm(corners):
    s3_vol[corner[0][0]:corner[1][0],corner[0][1]:corner[1][1],corner[0][2]:corner[1][2]] = local_vol[corner[0][0]:corner[1][0],corner[0][1]:corner[1][1],corner[0][2]:corner[1][2]]


