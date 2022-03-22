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

threshold = 0.5
chunk_size = [256, 256, 300]
dir = "s3://smartspim-precomputed-volumes/2022_01_14/8613/Ch_647"
data_dir = "/data/tathey1/matt_wright/brain_temp/"

coords = [0, 0]

print(f"Number cpus: {multiprocessing.cpu_count()}")

warnings.filterwarnings("ignore")

mip = 0
vol = CloudVolume(dir, parallel=True, mip=mip, fill_missing=True)
shape = vol.shape

corners = []
for i in tqdm(range(coords[0], shape[0], chunk_size[0])):
    for j in tqdm(range(coords[1], shape[1], chunk_size[1]), leave=False):  
        for k in range(0,shape[2],chunk_size[2]):
            c1 = [i,j,k]
            c2 = [np.amin([shape[idx], c1[idx]+chunk_size[idx]]) for idx in range(3)]
            corners.append([c1,c2])
    coords[1] = 0

def process_chunk(c1, c2, data_dir, threshold):
    mip = 0

    dir_base = "s3://smartspim-precomputed-volumes/2022_01_14/8613/"
    
    dir_mask = dir_base + "axon_mask"
    vol_mask = CloudVolume(dir_mask, parallel=1, mip=mip, fill_missing=True)

    dir_fg = dir_base + "Ch_647"
    vol_fg = CloudVolume(dir_fg, parallel=1, mip=mip, fill_missing=True)

    dir_bg = dir_base + "Ch_561"
    vol_bg = CloudVolume(dir_bg, parallel=1, mip=mip, fill_missing=True)

    dir_endo = dir_base + "Ch_488"
    vol_endo = CloudVolume(dir_endo, parallel=1, mip=mip, fill_missing=True)

    subvol_fg = np.squeeze(vol_fg[c1[0]:c2[0],c1[1]:c2[1],c1[1]:c2[1]])
    subvol_bg = np.squeeze(vol_bg[c1[0]:c2[0],c1[1]:c2[1],c1[1]:c2[1]])
    subvol_endo = np.squeeze(vol_endo[c1[0]:c2[0],c1[1]:c2[1],c1[1]:c2[1]])

    image_3channel = np.stack([subvol_bg, subvol_fg, subvol_endo], axis=0)

    fname = data_dir + "/image_" + str(c1[0]) + "_" + str(c1[1]) + "_" + str(c1[2]) + ".h5"
    with h5py.File(fname, "w") as f:
        dset = f.create_dataset("image_3channel", data=image_3channel)
    
    subprocess.run(["/home/tathey1/ilastik-1.3.3post3-Linux/run_ilastik.sh", "--headless", "--project=/data/tathey1/matt_wright/ilastik/model1/brain3/matt_benchmark_formal_brain3.ilp", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #subprocess.run(["/Applications/ilastik-1.3.3post3-OSX.app/Contents/ilastik-release/run_ilastik.sh", "--headless", "--project=/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain3/matt_benchmark_formal_brain3.ilp", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    fname_prob = fname[:-3] + "_Probabilities.h5"
    f = h5py.File(fname_prob, "r")
    pred = f.get("exported_data")

    pred = pred[1,:,:,:]

    mask = np.array(pred > threshold).astype('uint64')
    vol_mask[c1[0]:c2[0],c1[1]:c2[1],c1[1]:c2[1]] = mask

    print(f"Removing {fname} and {fname_prob}")
    os.remove(fname)
    os.remove(fname_prob)
    print(f"Finished {c1}-{c2}")


Parallel(n_jobs=-5)(delayed(process_chunk)(corner[0],corner[1], data_dir, threshold) for corner in tqdm(corners))
        

