'''
Inputs
'''
#s3 path to directory that contains the layers axon_mask (from segmentation), and atlas_to_target (from registration)
dir_base = "s3://smartspim-precomputed-volumes/2022_11_02/8786/" 
outdir = "/data/tathey1/matt_wright/brain_temp/" #directory to store temporary subvolumes for segmentation


'''
Collect results
'''
from sre_constants import CATEGORY_UNI_NOT_LINEBREAK
from tqdm import tqdm
import numpy as np
from cloudvolume import CloudVolume
import pickle
from joblib import Parallel, delayed
import os

if dir_base[-1] == "/":
    brain_id = dir_base.split("/")[-2]
else:
    brain_id = dir_base.split("/")[-1]


dir = os.path.join(dir_base, "axon_mask")
vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"Mask shape: {vol_mask.shape}")

dir = os.path.join(dir_base, "atlas_to_target")
vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"Atlas shape: {vol_reg.shape}")

block_size = [256, 256, 256]

corners = []
for x in tqdm(np.arange(0, vol_mask.shape[0], block_size[0])):
    x2 = np.amin([x + block_size[0], vol_mask.shape[0]])
    x_reg = int(x / 8)
    x2_reg = np.amin([int(x2 / 8), vol_reg.shape[0]])

    for y in tqdm(np.arange(0, vol_mask.shape[1], block_size[1]), leave=False):
        y2 = np.amin([y + block_size[1], vol_mask.shape[1]])
        y_reg = int(y / 8)
        y2_reg = np.amin([int(y2 / 8), vol_reg.shape[1]])
        for z in tqdm(np.arange(0, vol_mask.shape[2], block_size[2]), leave=False):
            z2 = np.amin([z + block_size[2], vol_mask.shape[2]])
            corners.append(
                [[x_reg, y_reg, z], [x2_reg, y2_reg, z2], [x, y, z], [x2, y2, z2]]
            )


def compute_composition_corner(corners, outdir, dir_base):
    l_c1 = corners[0]
    l_c2 = corners[1]
    m_c1 = corners[2]
    m_c2 = corners[3]

    fname = outdir + str(l_c1[0]) + "_" + str(l_c1[1]) + "_" + str(l_c1[2]) + ".pickle"
    if os.path.exists(fname):
        return

    dir = dir_base + "axon_mask"
    vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)

    dir = dir_base + "atlas_to_target"
    vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)

    labels = vol_reg[l_c1[0] : l_c2[0], l_c1[1] : l_c2[1], l_c1[2] : l_c2[2]]
    labels = np.repeat(np.repeat(labels, 8, axis=0), 8, axis=1)
    mask = vol_mask[m_c1[0] : m_c2[0], m_c1[1] : m_c2[1], m_c1[2] : m_c2[2]]

    width = np.amin([mask.shape[0], labels.shape[0]])
    height = np.amin([mask.shape[1], labels.shape[1]])
    mask = mask[:width, :height, :]
    labels = labels[:width, :height, :]

    labels_unique = np.unique(labels[labels > 0])

    volumes = {}
    for unq in labels_unique:
        cur_vol = np.sum(mask[labels == unq])
        cur_total = np.sum(labels == unq)
        volumes[unq] = [cur_total, cur_vol]

    with open(fname, "wb") as f:
        pickle.dump(volumes, f)


Parallel(n_jobs=-1)(
    delayed(compute_composition_corner)(corner, outdir, dir_base)
    for corner in tqdm(corners, desc="Finding labels")
)
# counter = 0
# for corner in corners:
#     if counter > 48:
#         raise ValueError()
#     l_c1 = corner[0]
#     fname = outdir + str(l_c1[0]) + "_" + str(l_c1[1]) + "_" + str(l_c1[2]) + ".pickle"
#     if not os.path.exists(fname):
#         print(fname)
#         counter += 1

files = os.listdir(outdir)

volumes = {}
for file in tqdm(files, desc="Assembling results"):
    if "pickle" in file:
        filename = outdir + file
        with open(filename, "rb") as f:
            result = pickle.load(f)
        for key in result.keys():
            addition = result[key]
            if key in volumes.keys():
                cur_vol = volumes[key][1]
                cur_total = volumes[key][0]
            else:
                cur_vol = 0
                cur_total = 0

            cur_vol += addition[1]
            cur_total += addition[0]
            volumes[key] = [cur_total, cur_vol]


outpath = outdir + "wholebrain_" + brain_id + ".pkl"
with open(outpath, "wb") as f:
    pickle.dump(volumes, f)
