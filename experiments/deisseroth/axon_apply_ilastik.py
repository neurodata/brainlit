import os 
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import numpy as np

model = "r1"

base_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/"
brains = ["8650", "8649", "8613", "8589", "8590", "8788"]
project_path= f"/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/axon_segmentation.ilp"
print(f"Number cpus: {multiprocessing.cpu_count()}")

def apply_ilastik(fname):
    if type(fname) == list:
        fname_str = ""
        for f in fname:
            if os.path.isfile(f) and ".h5" in f:
                fname_str += f
                fname_str += " "
            else:
                raise ValueError(f"Cannot process non-h5 file")
    else:
        if os.path.isfile(fname) and ".h5" in fname:
            fname_str = fname
        else:
            raise ValueError(f"Cannot process non-h5 file")

    subprocess.run(
        [
            "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh",
            "--headless",
            f"--project={project_path}",
            fname_str,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
def process_somas():
    items_total = []
    for brain in tqdm(brains, desc="Gathering brains..."):
        brain_name = brain

        path = f"{base_path}brain{brain_name}/"
        items = os.listdir(path)
        for item in items:
            item_path = path + item
            if os.path.isfile(item_path) and ".h5" in item_path and "Probabilities" not in item_path and "Labels" not in item_path:
                items_total.append(item_path)
    print(items_total)

    #run all files
    items_chunked = []
    if len(items_total) > 20:
        idxs = np.arange(0, len(items_total), 20)
        for idx in idxs:
            items_chunked.append(items_total[idx:idx+20])

    Parallel(n_jobs=8)(
        delayed(apply_ilastik)(
            item
        )
        for item in tqdm(items_total, desc="running ilastik...", leave=False)
    )

process_somas()
