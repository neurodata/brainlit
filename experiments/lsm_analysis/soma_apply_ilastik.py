"""
Inputs
"""
model = "-compare-r1-r2-878-887"
project_path = f"/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/matt_soma{model}.ilp"  # path to ilastik model to be used

base_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/"  # path to directory that holds images to be processed
brains = [
    "8607",
    "8606",
    "8477",
    "8531",
    "8608",
    "8529",
    "8557",
    "8555",
    "8446",
    "8454",
    "887",
]  # sample IDs to be processed

"""
Script
"""
import os
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing
from util import find_sample_names

print(f"Number cpus: {multiprocessing.cpu_count()}")


def apply_ilastik(fname):
    if os.path.isfile(fname) and ".h5" in fname:
        subprocess.run(
            [
                "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh",
                "--headless",
                f"--project={project_path}",
                fname,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )


def process_somas():
    items_total = []
    for brain in tqdm(brains, desc="Gathering brains..."):
        if brain == "8557":
            brain_name = "r1"
        elif brain == "8555":
            brain_name = "r2"
        else:
            brain_name = brain

        path = f"{base_path}brain{brain_name}/val/"

        items_total += find_sample_names(path, dset="", add_dir=True)

    # run all files
    Parallel(n_jobs=8)(
        delayed(apply_ilastik)(item)
        for item in tqdm(items_total, desc="running ilastik...")
    )


def move_results():
    # move results
    for brain in tqdm(brains, desc="Moving results"):
        if brain == "8557":
            brain_name = "r1"
        elif brain == "8555":
            brain_name = "r2"
        else:
            brain_name = brain

        brain_dir = f"{base_path}brain{brain_name}/val/"
        results_dir = brain_dir + "results" + model + "/"

        if not os.path.exists(results_dir):
            print(f"Creating directory: {results_dir}")
            os.makedirs(results_dir)

        items = find_sample_names(brain_dir, dset="", add_dir=False)
        for item in items:
            result_path = brain_dir + item[:-3] + "_Probabilities.h5"
            shutil.move(result_path, results_dir + item[:-3] + "_Probabilities.h5")


process_somas()
move_results()
