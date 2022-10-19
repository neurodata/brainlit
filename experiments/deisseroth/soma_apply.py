import os 
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing

base_path = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/"
brains = ["8607", "8606", "8477", "8531", "8608", "8529", "8557", "8555", "8446", "8454", "887"]


print(f"Number cpus: {multiprocessing.cpu_count()}")

def apply_ilastik(fname):
    if os.path.isfile(fname) and ".h5" in fname:
        subprocess.run(
            [
                "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh",
                "--headless",
                "--project=/Users/thomasathey/Documents/mimlab/mouselight/ailey/detection_soma/matt_soma_rabies_pix_3ch.ilp",
                fname,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

items_total = []
for brain in tqdm(brains, desc="Gathering brains..."):
    if brain == "8557":
        brain_name = "r1"
    elif brain == "8555":
        brain_name = "r2"
    else:
        brain_name = brain
    path = f"{base_path}brain{brain_name}/3channel/test/"
    items = os.listdir(path)

    for item in items:
        item_path = path + item
        if os.path.isfile(item_path) and ".h5" in item_path:
            os.remove(item_path)

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
    
Parallel(n_jobs=8)(
    delayed(apply_ilastik)(
        item
    )
    for item in tqdm(items_total, desc="running ilastik...", leave=False)
)
