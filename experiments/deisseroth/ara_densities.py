from tqdm import tqdm
import numpy as np
from cloudvolume import CloudVolume
import pickle

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/axon_mask"
vol_mask = CloudVolume(dir, parallel=0, mip=1, fill_missing=False)

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/atlas_to_target"
vol_reg = CloudVolume(dir, parallel=0, mip=1, fill_missing=False)

outdir = "/data/tathey1/matt_wright/brain4/vols_densities/"

volumes = {}
for x in tqdm(np.arange(0, vol_mask.shape[0], 128)):
    x2 = np.amin([x+128, vol_mask.shape[0]])
    for y in tqdm(np.arange(0, vol_mask.shape[1], 128), leave=False):
        y2 = np.amin([x+128, vol_mask.shape[1]])
        for z in tqdm(np.arange(0, vol_mask.shape[2], 128), leave=False):
            z2 = np.amin([x+128, vol_mask.shape[2]])
            labels = vol_reg[x:x2,y:y2,z:z2]
            labels_unique = np.unique(labels)
            mask = vol_mask[x:x2,y:y2,z:z2]

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