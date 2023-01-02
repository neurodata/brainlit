'''
Inputs
'''
model = "-compare-3_4"
project_path= f"/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/axon_segmentation{model}.ilp" #path to ilastik model to be used

base_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/" #path to directory that holds images to be processed
brains = ["8650", "8649", "8613", "8589", "8590", "8788"] #sample IDs to be processed

'''
Script
'''
import os 
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from util import find_sample_names


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



def process_images():
    items_total = []
    for brain in tqdm(brains, desc="Gathering brains..."):
        path = f"{base_path}brain{brain}/"
        items_total += find_sample_names(path, dset = "val", add_dir=True)

    print(items_total)

    Parallel(n_jobs=8)(
        delayed(apply_ilastik)(
            item
        )
        for item in tqdm(items_total, desc="running ilastik...", leave=False)
    )

def move_results():
    for brain in tqdm(brains, desc="Moving results..."):
        # make results folder
        brain_dir = f"{base_path}brain{brain}/"
        results_dir = f"{brain_dir}/results{model}/"

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        items = find_sample_names(brain_dir, dset = "val", add_dir=False)
        for item in items:
            result_path = brain_dir + item[:-3] + "_Probabilities.h5"
            shutil.move(result_path, results_dir+item[:-3] + "_Probabilities.h5")


process_images()
move_results()