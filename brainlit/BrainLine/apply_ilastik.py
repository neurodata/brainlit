import os
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing
from brainlit.BrainLine.util import find_sample_names
from brainlit.BrainLine.data.soma_data import brain2paths
from datetime import date
from cloudvolume import CloudVolume, exceptions
import numpy as np
import h5py
from skimage import io, measure
from pathlib import Path
from os.path import isfile, join
from os import listdir
import random
from cloudreg.scripts.transform_points import NGLink
from cloudreg.scripts.visualization import create_viz_link_from_json


class ApplyIlastik:
    def __init__(self, ilastk_path: str, project_path: str, brains_path: str, brains: list):
        self.ilastk_path = ilastk_path
        self.project_path = project_path
        self.brains_path = brains_path
        self.brains = brains

    def apply_ilastik(self, fname):
        if os.path.isfile(fname) and ".h5" in fname:
            subprocess.run(
                [
                    "/Applications/ilastik-1.4.0b21-OSX.app/Contents/ilastik-release/run_ilastik.sh",
                    "--headless",
                    f"--project={self.project_path}",
                    fname,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

    def process_somas(self):
        items_total = []
        for brain in tqdm(self.brains, desc="Gathering brains..."):

            path = f"{self.brains_path}brain{brain}/val/"

            items_total += find_sample_names(path, dset="", add_dir=True)

        # run all files
        Parallel(n_jobs=8)(
            delayed(self.apply_ilastik)(item)
            for item in tqdm(items_total, desc="running ilastik...")
        )

    def move_results(self):
        # move results
        for brain in tqdm(self.brains, desc="Moving results"):
            # if brain == "8557":
            #     brain_name = "r1"
            # elif brain == "8555":
            #     brain_name = "r2"
            # else:
            #     brain_name = brain

            brain_dir = f"{self.brains_path}brain{brain}/val/"
            results_dir = brain_dir + "results" + str(date.today()) + "/"

            if not os.path.exists(results_dir):
                print(f"Creating directory: {results_dir}")
                os.makedirs(results_dir)

            items = find_sample_names(brain_dir, dset="", add_dir=False)
            for item in items:
                result_path = brain_dir + item[:-3] + "_Probabilities.h5"
                shutil.move(result_path, results_dir + item[:-3] + "_Probabilities.h5")

class ApplyIlastik_LargeImage:
    def __init__(self, ilastik_path: str, ilastik_project: str, results_dir: str, ncpu: int):
        self.ilastik_path = ilastik_path
        self.ilastik_project = ilastik_project
        self.results_dir = results_dir
        self.ncpu = ncpu
        

    def apply_ilastik_parallel(self, brain_id: str, layer_names: list, threshold: float, data_dir: str, chunk_size: list, max_coords: list = [-1, -1, -1]):
        results_dir = self.results_dir

        isExist = os.path.exists(data_dir)
        if not isExist:
            print(f"Creating directory: {data_dir}")
            os.makedirs(data_dir)
        else:
            print(f"Downloaded data will be stored in {data_dir}")
        isExist = os.path.exists(results_dir)
        if not isExist:
            print(f"Creating directory: {results_dir}")
            os.makedirs(results_dir)
        else:
            print(f"Downloaded data will be stored in {results_dir}")

        volume_base_dir = brain2paths[brain_id]["base"]
        sample_path = volume_base_dir + layer_names[0]
        vol = CloudVolume(sample_path, parallel=True, mip=0, fill_missing=True)
        shape = vol.shape
        print(f"Processing: {sample_path} with shape {shape} at threshold {threshold}")

        corners = self._get_corners(shape, chunk_size, max_coords)
        corners_chunks = [corners[i : i + 100] for i in range(0, len(corners), 100)]

        for corners_chunk in tqdm(corners_chunks, desc="corner chunks"):
            results = Parallel(n_jobs=self.ncpu)(
                delayed(self.process_chunk)(
                    corner[0], corner[1], volume_base_dir, layer_names, threshold, data_dir, results_dir
                )
                for corner in tqdm(corners_chunk, leave=False)
            )
            for f in os.listdir(data_dir):
                os.remove(os.path.join(data_dir, f))


    def _get_corners(self, shape, chunk_size, max_coords):
        corners = []
        for i in tqdm(range(0, shape[0], chunk_size[0])):
            for j in tqdm(range(0, shape[1], chunk_size[1]), leave=False):
                for k in range(0, shape[2], chunk_size[2]):
                    c1 = [i, j, k]
                    c2 = [np.amin([shape[idx], c1[idx] + chunk_size[idx]]) for idx in range(3)]
                    conditions = [(max == -1 or c < max) for c,max in zip(c1, max_coords)]
                    if all(conditions):
                        corners.append([c1, c2])

        return corners

    def process_chunk(self, c1: list, c2: list, volume_base_dir: str, layer_names: list, threshold: float, data_dir: str, results_dir: str):
        mip = 0
        area_threshold = 500

        dir_fg = volume_base_dir + layer_names[0]
        vol_fg = CloudVolume(dir_fg, parallel=1, mip=mip, fill_missing=False)
        dir_bg = volume_base_dir + layer_names[1]
        vol_bg = CloudVolume(dir_bg, parallel=1, mip=mip, fill_missing=True)
        dir_endo = volume_base_dir + layer_names[2]
        vol_endo = CloudVolume(dir_endo, parallel=1, mip=mip, fill_missing=True)

        try:
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
        except exceptions.EmptyVolumeException:
            return


        fname = (
            data_dir + "image_" + str(c1[0]) + "_" + str(c1[1]) + "_" + str(c1[2]) + ".h5"
        )
        with h5py.File(fname, "w") as f:
            dset = f.create_dataset("image_3channel", data=image_3channel)

        subprocess.run(
            [
                f"{self.ilastik_path}",
                "--headless",
                f"--project={self.ilastik_project}",
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
                    location = list(np.add(c1, prop["centroid"]))
                    results.append(location)
            if len(results) > 0:
                with open(fname_results, "w") as f2:
                    for location in results:
                        f2.write(str(location))
                        f2.write("\n")

    def collect_results(self, brain_id: str):
        coords = []
        coords_target_space = []
        results_dir = self.results_dir
        onlyfiles = [join(results_dir, f) for f in listdir(results_dir) if isfile(join(results_dir, f))]
        onlyfiles = [f for f in onlyfiles if ".txt" in f]
        div_factor = [8, 8, 1]
        for file in tqdm(onlyfiles, desc="reading files"):
            print(file)
            file1 = open(file, "r")
            lines = file1.readlines()

            for line in tqdm(lines, desc="parsing coordinates", leave=False):
                if line != "\n":
                    line = " ".join(line.split())
                    elements = line.split(",")
                    coord = [elements[0][1:], elements[1], elements[2][:-1]]

                    coords_target_space.append([float(e.strip()) for e in coord])
                    coord = [
                        int(round(float(e.strip()) / f)) for e, f in zip(coord, div_factor)
                    ]
                    coords.append(coord)   
        print(f"{len(coords)} somas detected, first is: {coords_target_space[0]}")
        all_somas_path = results_dir + "all_somas_" + brain_id + ".txt"
        print(f"Writing {all_somas_path}...")
        with open(all_somas_path, "w") as f:
            for coord in coords_target_space:
                f.write(f"{coord}")
                f.write("\n")

        if len(coords_target_space) > 2000:
            random.shuffle(coords_target_space)
            coords_target_space = coords_target_space[:2000]
            print("*********Only posting first 2000 somas to neuroglancer**********")
            name = "detected_somas_partial"
        else:
            name = "detected_somas"


        ng_link = brain2paths[brain_id]["val_info"]["url"]
        viz_link = NGLink(ng_link.split("json_url=")[-1])
        ngl_json = viz_link._json

        ngl_json["layers"] = [layer for layer in ngl_json["layers"] if layer["type"] != "annotation"]


        ngl_json["layers"].append(
            {"type": "annotation", "points": coords_target_space, "name": name}
        )
        viz_link = create_viz_link_from_json(
            ngl_json, neuroglancer_link="https://viz.neurodata.io/?json_url="
        )
        print(f"Viz link with detections: {viz_link}")