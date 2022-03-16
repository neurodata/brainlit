from tqdm import tqdm
import numpy as np
from cloudvolume import CloudVolume
import pickle

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/axon_mask"
vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"Mask shape: {vol_mask.shape}")

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/atlas_to_target"
vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"Atlas shape: {vol_reg.shape}")

outdir = "/data/tathey1/matt_wright/brain4/vols_densities/"
#outdir = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain4"

volumes = {}
for x in tqdm(np.arange(0, vol_mask.shape[0], 128)):
    x2 = np.amin([x+128, vol_mask.shape[0]])
    x_reg = int(x/8)
    x2_reg = np.amin([int(x2/8), vol_reg.shape[0]])

    for y in tqdm(np.arange(0, vol_mask.shape[1], 128), leave=False):
        y2 = np.amin([y+128, vol_mask.shape[1]])
        y_reg = int(y/8)
        y2_reg = np.amin([int(y2/8), vol_reg.shape[1]])
        for z in tqdm(np.arange(0, vol_mask.shape[2], 128), leave=False):
            z2 = np.amin([z+128, vol_mask.shape[2]])
            labels = vol_reg[x_reg:x2_reg,y_reg:y2_reg,z:z2]

            mask = vol_mask[x:x2,y:y2,z:z2]

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

    if x % 512 == 0:
        outpath = outdir + str(x/512) + ".pkl"
        with open(outpath, 'wb') as f:
            pickle.dump(volumes, f)

outpath = outdir + "vol_density.pkl"
with open(outpath, 'wb') as f:
    pickle.dump(volumes, f)