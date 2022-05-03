from cloudvolume import CloudVolume
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

chunk_size = [256, 256, 300]
ncpu = 16
dir_base = "precomputed://s3://smartspim-precomputed-volumes/2022_03_15/8606/"
data_dir = "/data/tathey1/matt_wright/brainr_temp/"
results_dir = "/data/tathey1/matt_wright/brainr_results/"
threshold = 0.6

print(f"Number cpus: {multiprocessing.cpu_count()}")

warnings.filterwarnings("ignore")


def process_chunk(c1, c2, dir_base, threshold, data_dir, results_dir):
    chunk_size = [256, 256, 300]
    mip = 0
    area_threshold = 500

    dir_fg = dir_base + "Ch_647"
    vol_fg = CloudVolume(dir_fg, parallel=1, mip=mip, fill_missing=True)
    dir_bg = dir_base + "Ch_561"
    vol_bg = CloudVolume(dir_bg, parallel=1, mip=mip, fill_missing=True)
    dir_endo = dir_base + "Ch_488"
    vol_endo = CloudVolume(dir_endo, parallel=1, mip=mip, fill_missing=True)

    shape = vol_fg.shape

    image_3channel = np.squeeze(
        np.stack(
            [
                vol_fg[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]],
                vol_bg[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]],
                vol_endo[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]],
            ],
            axis=0,
        )
    )

    fname = (
        data_dir + "image_" + str(c1[0]) + "_" + str(c1[1]) + "_" + str(c1[2]) + ".h5"
    )
    with h5py.File(fname, "w") as f:
        dset = f.create_dataset("image_3channel", data=image_3channel)

    subprocess.run(
        [
            "/home/tathey1/ilastik-1.3.3post3-Linux/run_ilastik.sh",
            "--headless",
            "--project=/data/tathey1/matt_wright/ilastik/soma_model/matt_soma_rabies_pix_3ch.ilp",
            fname,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # subprocess.run(["/Applications/ilastik-1.3.3post3-OSX.app/Contents/ilastik-release/run_ilastik.sh", "--headless", "--project=/Users/thomasathey/Documents/mimlab/mouselight/ailey/benchmark_formal/brain3/matt_benchmark_formal_brain3.ilp", fname], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    fname_prob = fname[:-3] + "_Probabilities.h5"
    fname_results = (
        results_dir
        + "image_"
        + str(c1[0])
        + "_"
        + str(c1[1])
        + "_"
        + str(c1[2])
        + "_somas.txt"
    )
    with h5py.File(fname_prob, "r") as f:
        pred = f.get("exported_data")
        pred = pred[0, :, :, :]
        mask = pred > threshold
        labels = measure.label(mask)
        props = measure.regionprops(labels)

        results = []
        for prop in props:
            if prop["area"] > area_threshold:
                location = list(np.add((i, j, k), prop["centroid"]))
                results.append(location)
        if len(results) > 0:
            with open(fname_results, "w") as f2:
                for location in results:
                    f2.write(str(location))
                    f2.write("\n")


mip = 0
sample_path = dir_base + "Ch_647"
vol = CloudVolume(sample_path, parallel=True, mip=mip, fill_missing=True)
shape = vol.shape
print(f"Processing: {sample_path} with shape {shape} at threshold {threshold}")


corners = []
for i in tqdm(range(0, shape[0], chunk_size[0])):
    if i < 1500:
        continue
    for j in tqdm(range(0, shape[1], chunk_size[1]), leave=False):
        for k in range(0, shape[2], chunk_size[2]):
            c1 = [i, j, k]
            c2 = [np.amin([shape[idx], c1[idx] + chunk_size[idx]]) for idx in range(3)]
            corners.append([c1, c2])

corners_chunks = [corners[i : i + 100] for i in range(0, len(corners), 100)]

for corners_chunk in tqdm(corners_chunks, desc="corner chunks"):
    results = Parallel(n_jobs=ncpu)(
        delayed(process_chunk)(
            corner[0], corner[1], dir_base, threshold, data_dir, results_dir
        )
        for corner in tqdm(corners_chunk, leave=False)
    )
    for f in os.listdir(data_dir):
        os.remove(os.path.join(data_dir, f))
