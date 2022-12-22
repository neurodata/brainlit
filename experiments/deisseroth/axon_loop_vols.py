'''
Inputs
'''
dir_base = "precomputed://s3://smartspim-precomputed-volumes/2022_03_28/8649/" #s3 path to directory that contains image data
threshold = 0.4 #threshold to use for ilastik
data_dir = "/data/tathey1/matt_wright/brain_temp/" #directory to store temporary subvolumes for segmentation
max_y = -1 #maxy coord, or -1 if you want to process all of them
skip_segment = False

'''
Segment Axon
'''
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

chunk_size = [256, 256, 300]

print(f"***********DID YOU REMEMBER TO UPDATE THE THRESHOLD**********")
print(f"Number cpus: {multiprocessing.cpu_count()}")

warnings.filterwarnings("ignore")

mip = 0
vol = CloudVolume(dir_base + "Ch_647", parallel=True, mip=mip, fill_missing=True)
shape = list(vol.shape)

corners = []
for i in tqdm(range(0, shape[0], chunk_size[0])):
    for j in tqdm(range(0, shape[1], chunk_size[1]), leave=False):
        for k in range(0, shape[2], chunk_size[2]):
            c1 = [i, j, k]
            c2 = [np.amin([shape[idx], c1[idx] + chunk_size[idx]]) for idx in range(3)]
            if max_y == -1 or c1[1] < max_y:
                corners.append([c1, c2])

corners_chunks = [corners[i : i + 100] for i in range(0, len(corners), 100)]

print(f"Processing brain of size {vol.shape}")


def process_chunk(c1, c2, data_dir, threshold, dir_base):
    mip = 0

    dir_mask = dir_base + "axon_mask"
    vol_mask = CloudVolume(dir_mask, parallel=1, mip=mip, fill_missing=True)

    dir_fg = dir_base + "Ch_647"
    vol_fg = CloudVolume(dir_fg, parallel=1, mip=mip, fill_missing=True)

    dir_bg = dir_base + "Ch_561"
    vol_bg = CloudVolume(dir_bg, parallel=1, mip=mip, fill_missing=True)

    dir_endo = dir_base + "Ch_488"
    vol_endo = CloudVolume(dir_endo, parallel=1, mip=mip, fill_missing=True)

    subvol_fg = np.squeeze(vol_fg[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]])
    subvol_bg = np.squeeze(vol_bg[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]])
    subvol_endo = np.squeeze(vol_endo[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]])

    image_3channel = np.stack([subvol_bg, subvol_fg, subvol_endo], axis=0)

    fname = (
        data_dir + "image_" + str(c1[0]) + "_" + str(c1[1]) + "_" + str(c1[2]) + ".h5"
    )
    with h5py.File(fname, "w") as f:
        dset = f.create_dataset("image_3channel", data=image_3channel)

    subprocess.run(
        [
            "/data/tathey1/matt_wright/ilastik/ilastik-1.4.0rc5-Linux/run_ilastik.sh",
            "--headless",
            "--project=/data/tathey1/matt_wright/ilastik/model1/axon_segmentation.ilp",
            fname,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # subprocess.run(["/Applications/ilastik-1.3.3post3-OSX.app/Contents/ilastik-release/run_ilastik.sh", "--headless", "--project=/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain3/matt_benchmark_formal_brain3.ilp", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    fname_prob = fname[:-3] + "_Probabilities.h5"
    with h5py.File(fname_prob, "r") as f:
        pred = f.get("exported_data")
        pred = pred[1, :, :, :]
        mask = np.array(pred > threshold).astype("uint64")
        vol_mask[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]] = mask

if not skip_segment:
    for corners_chunk in tqdm(corners_chunks, desc="corner chunks"):
        # for corner in tqdm(corners_chunk):
        #      process_chunk(corner[0],corner[1], data_dir, threshold, dir_base)
        Parallel(n_jobs=15)(
            delayed(process_chunk)(corner[0], corner[1], data_dir, threshold, dir_base)
            for corner in tqdm(corners_chunk, leave=False)
        )
        for f in os.listdir(data_dir):
            os.remove(os.path.join(data_dir, f))


'''
Downsample Mask
'''
print("Downsampling...")
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
    delete_black_uploads=False,  # issue a delete instead of uploading files containing all background
    background_color=0,  # Designates the background color
    compress="gzip",  # None, 'gzip', and 'br' (brotli) are options
    factor=(2, 2, 2),  # common options are (2,2,1) and (2,2,2)
)

tq.insert(tasks)
tq.execute()

'''
Make transformed layer
'''
layer_path = dir_base + "axon_mask_transformed"

atlas_vol = CloudVolume("precomputed://https://open-neurodata.s3.amazonaws.com/ara_2016/sagittal_10um/annotation_10um_2017")

info = CloudVolume.create_new_info(
    num_channels=1,
    layer_type="image",
    data_type="uint8",  # Channel images might be 'uint8'
    encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
    resolution=atlas_vol.resolution,  # Voxel scaling, units are in nanometers
    voxel_offset=atlas_vol.voxel_offset,  # x,y,z offset in voxels from the origin
    # mesh            = 'mesh',
    # Pick a convenient size for your underlying chunk representation
    # Powers of two are recommended, doesn't need to cover image exactly
    chunk_size=[32, 32, 32],  # units are voxels
    volume_size=atlas_vol.volume_size,  # e.g. a cubic millimeter dataset
)
vol_mask = CloudVolume(layer_path, info=info)
vol_mask.commit_info()