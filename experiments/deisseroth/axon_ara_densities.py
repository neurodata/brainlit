from sre_constants import CATEGORY_UNI_NOT_LINEBREAK
from tqdm import tqdm
import numpy as np
from cloudvolume import CloudVolume
import pickle
from joblib import Parallel, delayed

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/axon_mask"
vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"Mask shape: {vol_mask.shape}")

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/atlas_to_target"
vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"Atlas shape: {vol_reg.shape}")

outdir = "/data/tathey1/matt_wright/brain4/vols_densities/"
#outdir = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain4/"

n_jobs = 24

corners = []
for x in tqdm(np.arange(2816, vol_mask.shape[0], 128)):
    x2 = np.amin([x+128, vol_mask.shape[0]])
    x_reg = int(x/8)
    x2_reg = np.amin([int(x2/8), vol_reg.shape[0]])

    for y in tqdm(np.arange(0, vol_mask.shape[1], 128), leave=False):
        y2 = np.amin([y+128, vol_mask.shape[1]])
        y_reg = int(y/8)
        y2_reg = np.amin([int(y2/8), vol_reg.shape[1]])
        for z in tqdm(np.arange(0, vol_mask.shape[2], 128), leave=False):
            z2 = np.amin([z+128, vol_mask.shape[2]])

            corners.append([[x_reg, y_reg, z], [x2_reg, y2_reg, z2], [x,y,z], [x2,y2,z2]])

block_size = int(np.ceil(len(corners)/n_jobs))
corners_blocks = [corners[i:i+block_size] for i in range(0, len(corners), block_size)]
print(f"corners len: {len(corners)}")
sum_len = np.sum([len(block) for block in corners_blocks])
print(f"{len(corners_blocks)} blocks with total len {sum_len}")


def compute_composition_block(corners_block):
    dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/axon_mask"
    vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)

    dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/atlas_to_target"
    vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)

    volumes = {}
    for corners in tqdm(corners_block, desc="going through blocks", leave=False):
        l_c1 = corners[0]
        l_c2 = corners[1]
        m_c1 = corners[2]
        m_c2 = corners[3]

        labels = vol_reg[l_c1[0]:l_c2[0],l_c1[1]:l_c2[1],l_c1[2]:l_c2[2]]
        labels = np.repeat(np.repeat(labels, 8, axis=0), 8, axis=1)
        mask = vol_mask[m_c1[0]:m_c2[0],m_c1[1]:m_c2[1],m_c1[2]:m_c2[2]]

        width = np.amin([mask.shape[0], labels.shape[0]])
        height = np.amin([mask.shape[1], labels.shape[1]])
        mask = mask[:width, :height, :]
        labels = labels[:width, :height, :]

        labels_unique = np.unique(labels[labels > 0])

        for unq in labels_unique:
            if unq in volumes.keys():
                cur_vol = volumes[unq][1]
                cur_total = volumes[unq][0]
            else:
                cur_vol = 0
                cur_total = 0
            cur_vol += np.sum(mask[labels == unq])
            cur_total += np.sum(labels == unq)
            volumes[unq] = [cur_total, cur_vol]
    return volumes


results = Parallel(n_jobs=n_jobs)(delayed(compute_composition_block)(corners) for corners in corners_blocks)

volumes = {}
for result in tqdm(results, desc="Assembling results"):
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


outpath = outdir + "vol_density.pkl"
with open(outpath, 'wb') as f:
    pickle.dump(volumes, f)