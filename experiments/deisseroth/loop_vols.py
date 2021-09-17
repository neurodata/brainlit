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

chunk_size = [256, 256, 300]
ncpu = 12
dir = "s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_647"
progress_file = "/home/tathey1/progress.txt" #"/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain4/tracing/progress.txt" 
files_dir = "/data/tathey1/matt_wright/brain4/tracing/"

with open(progress_file) as f:
    for line in f:
        pass
    last_line = line

coords_list = last_line.split(' ')
coords = [int(coord) for coord in coords_list]

print(f"Number cpus: {multiprocessing.cpu_count()}")

warnings.filterwarnings("ignore")


def process_chunk(i, j, k):
    data_dir = "/data/tathey1/matt_wright/brain4/tracing/"
    #data_dir = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain4/tracing/"

    chunk_size = [256, 256, 300]
    mip = 0
    
    dir_mask = "s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/axon_mask"
    vol_mask = CloudVolume(dir_mask, parallel=1, mip=mip, fill_missing=True)

    dir_fg = "s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_647"
    vol_fg = CloudVolume(dir_fg, parallel=1, mip=mip, fill_missing=True)

    dir_bg = "s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_561"
    vol_bg = CloudVolume(dir_bg, parallel=1, mip=mip, fill_missing=True)

    dir_endo = "s3://smartspim-precomputed-volumes/2021_07_15_Sert_Cre_R/Ch_488"
    vol_endo = CloudVolume(dir_endo, parallel=1, mip=mip, fill_missing=True)

    shape = vol_fg.shape

    i2 = np.amin([i+chunk_size[0], shape[0]])
    j2 = np.amin([j+chunk_size[1], shape[1]])
    k2 = np.amin([k+chunk_size[2], shape[2]])
    subvol_fg = np.squeeze(vol_fg[i:i2,j:j2,k:k2])
    subvol_bg = np.squeeze(vol_bg[i:i2,j:j2,k:k2])
    subvol_endo = np.squeeze(vol_endo[i:i2,j:j2,k:k2])

    image_3channel = np.stack([subvol_bg, subvol_fg, subvol_endo], axis=0)

    fname = data_dir + "/image_" + str(k) + ".h5"
    with h5py.File(fname, "w") as f:
        dset = f.create_dataset("image_3channel", data=image_3channel)
    
    subprocess.run(["/home/tathey1/ilastik-1.3.3post3-Linux/run_ilastik.sh", "--headless", "--project=/data/tathey1/matt_wright/ilastik/model1/matt_benchmark_formal_brain3.ilp", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #subprocess.run(["/Applications/ilastik-1.3.3post3-OSX.app/Contents/ilastik-release/run_ilastik.sh", "--headless", "--project=/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain3/matt_benchmark_formal_brain3.ilp", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    f = h5py.File(data_dir + "image_" + str(k) + "_Probabilities.h5", "r")
    pred = f.get("exported_data")

    pred = pred[1,:,:,:]

    mask = np.array(pred > 0.5).astype('uint64')
    print(f"{k}: {np.sum(mask)}, {mask.shape}")
    vol_mask[i:i2,j:j2,k:k2] = mask

mip = 0
vol = CloudVolume(dir, parallel=True, mip=mip, fill_missing=True)
shape = vol.shape


for i in tqdm(range(coords[0], shape[0], chunk_size[0])):
    for j in tqdm(range(coords[1], shape[1], chunk_size[1]), leave=False):    
        Parallel(n_jobs=ncpu)(delayed(process_chunk)(i,j,k) for k in range(0,shape[2],chunk_size[2]))
        with open(progress_file, 'a') as f:
            f.write('\n')
            f.write(f'{i} {j}')
        for f in os.listdir(files_dir):
            os.remove(os.path.join(dir, f))
        raise ValueError()

