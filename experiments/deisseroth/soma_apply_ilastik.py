import os 
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing

model = "r1_r2"

base_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/"
brains = ["8607", "8606", "8477", "8531", "8608", "8529", "8557", "8555", "8446", "8454", "887"]
project_path= f"/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/matt_soma{model}.ilp"
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

        path = f"{base_path}brain{brain_name}/3channel/test/"
        # delete files already there
        items = os.listdir(path)

        for item in items:
            item_path = path + item
            if os.path.isfile(item_path) and ".h5" in item_path:
                os.remove(item_path)

        # copy images
        path_images_only = path + "images_only/"
        items = os.listdir(path_images_only)

        for item in items:
            if ".h5" in item:
                src = path_images_only + item
                dst = path + item
                shutil.copyfile(src, dst)


        items = os.listdir(path)
        items = [path + i for i in items]
        items_total += items
        
    #run all files
    Parallel(n_jobs=8)(
        delayed(apply_ilastik)(
            item
        )
        for item in tqdm(items_total, desc="running ilastik...", leave=False)
    )

    # move results
    for brain in tqdm(brains, desc="Moving results"):
        if brain == "8557":
            brain_name = "r1"
        elif brain == "8555":
            brain_name = "r2"
        else:
            brain_name = brain

        path_base = f"{base_path}brain{brain_name}/3channel/test/"
        path_results = path_base + "results" + model + "/"

        isExist = os.path.exists(path_results)
        if not isExist:
            print(f"Creating directory: {path_results}")
            os.makedirs(path_results)

        # delete files already there
        files = os.listdir(path_base)

        for f in files:
            if "Probabilities.h5" in f:
                src = path_base + f
                dst = path_results + f
                shutil.copyfile(src, dst)

process_somas()