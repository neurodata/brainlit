"""
Inputs
"""
models = ["-compare-3","-compare-3-4","-compare-3-4-8649","-compare-3-4-8649-8788","-compare-3_2","-compare-3_3","-compare-3_4"]

base_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/"  # path to directory that holds images to be processed
brains = ["8786", "8790", "11537"]  # sample IDs to be processed

"""
Script
"""
import os
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from util import find_sample_names


print(f"Number cpus: {multiprocessing.cpu_count()}")


def apply_ilastik(fname, project_path):
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


def process_images(model):
    project_path = f"/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_axon/axon_segmentation{model}.ilp"
    items_total = []
    for brain in tqdm(brains, desc="Gathering brains..."):
        path = f"{base_path}brain{brain}/"
        items_total += find_sample_names(path, dset="val", add_dir=True)

    print(items_total)

    Parallel(n_jobs=8)(
        delayed(apply_ilastik)(item, project_path)
        for item in tqdm(items_total, desc="running ilastik...", leave=False)
    )


def move_results(model):
    for brain in tqdm(brains, desc="Moving results..."):
        # make results folder
        brain_dir = f"{base_path}brain{brain}/"
        results_dir = f"{brain_dir}/results{model}/"

        if not os.path.exists(results_dir):
            os.mkdir(results_dir)

        items = find_sample_names(brain_dir, dset="val", add_dir=False)
        for item in items:
            result_path = brain_dir + item[:-3] + "_Probabilities.h5"
            shutil.move(result_path, results_dir + item[:-3] + "_Probabilities.h5")

for model in models:
    process_images(model)
    move_results(model)
