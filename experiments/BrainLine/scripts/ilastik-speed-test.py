import pandas as pd
import numpy as np
import os
from pathlib import Path
import h5py
import subprocess
import time
from tqdm import tqdm

project_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/axon_segmentation.ilp"  # path to ilastik model to be used
ilastik_path = (
    "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh"
)
data_bin = Path(
    "/Users/thomasathey/Documents/mimlab/mouselight/brainlit_parent/brainlit/experiments/BrainLine/data/ilastik-timing/"
)

# multiple images vs. multiple calls
# vary chunking

if not os.path.exists(data_bin):
    os.mkdir(data_bin)

times = []
commands = []
files = []
image_sizes = []

for radius in tqdm([2**i for i in range(8, 11)], desc="different file sizes..."):
    data = np.random.normal(size=(3, radius, radius, radius))
    for num_splits in tqdm(
        [2**i for i in range(1, 3)], desc="different chunkings...", leave=False
    ):
        num_files = num_splits**3

        split_ras = [data]
        for axis in range(1, 4):
            split_ras = [
                sub_ra
                for ra in split_ras
                for sub_ra in np.split(ra, num_splits, axis=axis)
            ]

        fnames = []
        for i, ra in enumerate(split_ras):
            fname = data_bin / f"r{radius}-n{num_files}-i{i}.h5"
            fnames.append(str(fname))
            with h5py.File(fname, "w") as f:
                dset = f.create_dataset("image_3channel", data=ra)

        command = [
            f"{ilastik_path}",
            "--headless",
            f"--project={project_path}",
        ] + fnames

        t0 = time.time()
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        delta = time.time() - t0

        times.append(delta)
        image_sizes.append(data.itemsize * data.size)
        files.append(num_files)
        commands.append("Single")

        t0 = time.time()
        for fname in fnames:
            command = [
                f"{ilastik_path}",
                "--headless",
                f"--project={project_path}",
                fname,
            ]
            subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        delta = time.time() - t0

        times.append(delta)
        image_sizes.append(data.itemsize * data.size)
        files.append(num_files)
        commands.append("Multiple")

        for f in os.listdir(data_bin):
            os.remove(data_bin / f)

data = {
    "Command": commands,
    "Number of Files": files,
    "Overall Image Size": image_sizes,
    "Time": times,
}
df = pd.DataFrame(data=data)

df.to_csv(data_bin / "results.csv")
