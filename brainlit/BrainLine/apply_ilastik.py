import os
import shutil
from tqdm import tqdm
import subprocess
from joblib import Parallel, delayed
import multiprocessing
from brainlit.BrainLine.util import _find_sample_names, _get_corners
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
import pandas as pd
from brainlit.BrainLine.imports import *
import json
from typing import Union


class ApplyIlastik:
    """Applies ilastik to subvolumes for the purpose of validating machine learning algorithms.

    Arguments:
        ilastk_path (str): Path to ilastik executable.
        project_path (str): Path to ilastik project.
        brains_path (str): Path to directory that contains brain samples subdirectories.
        brains (list): List of brain sample names.

    Attributes:
        ilastk_path (str): Path to ilastik executable.
        project_path (str): Path to ilastik project.
        brains_path (str): Path to directory that contains brain samples subdirectories.
        brains (list): List of brain sample names.

    """

    def __init__(
        self, ilastk_path: str, project_path: str, brains_path: str, brains: list
    ):
        self.ilastk_path = ilastk_path
        self.project_path = project_path
        self.brains_path = brains_path
        self.brains = brains

    def _apply_ilastik(self, fname):
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

    def process_subvols(self):
        """Apply ilastik to all validation subvolumes of the specified brain ids in the specified directory"""
        items_total = []
        for brain in tqdm(self.brains, desc="Gathering brains..."):
            path = f"{self.brains_path}brain{brain}/val/"

            items_total += _find_sample_names(path, dset="", add_dir=True)

        # run all files
        Parallel(n_jobs=8)(
            delayed(self._apply_ilastik)(item)
            for item in tqdm(items_total, desc="running ilastik...")
        )

    def move_results(self):
        """Move results from process_subvols to a new subfolder."""
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

            items = _find_sample_names(brain_dir, dset="", add_dir=False)
            for item in items:
                result_path = brain_dir + item[:-3] + "_Probabilities.h5"
                shutil.move(result_path, results_dir + item[:-3] + "_Probabilities.h5")


def plot_results(
    data_dir: str,
    brain_ids: list,
    object_type: str,
    positive_channel: int,
    doubles: list = [],
    show_plot: bool = True,
):
    """Plot precision recall curve for a specified brain.

    Args:
        data_dir (str): Path to directory where brain subvolumes are stored.
        brain_id (str): Brain id to examine (brain2paths key from _data.py file).
        object_type (str): soma or axon, the type of data to examine.
        positive_channel (int): Channel that represents neuron in the predictions.
        doubles (list, optional): Filenames of soma subvolumes that contain two somas, if applicable. Defaults to [].
        show_plot (bool, optional): Whether to run pyplot, useful for pytests when figures should not be displayed. Defaults to True.

    Raises:
        ValueError: _description_

    Returns:
        float: Best f-score across all thresholds.
        float: Threshold that yields the best validation f-score.
    """
    recalls = []
    precisions = []
    brain_ids_data = []
    best_fscores = {}
    best_precisions = []
    best_recalls = []

    size_thresh = 500

    thresholds = list(np.arange(0.0, 1.0, 0.02))
    for brain_id in tqdm(brain_ids, desc="Processing Brains"):
        base_dir = data_dir + f"/brain{brain_id}/val/"
        data_files = _find_sample_names(base_dir, dset="", add_dir=True)
        test_files = [file.split(".")[0] + "_Probabilities.h5" for file in data_files]

        best_fscore = 0
        best_thresh = -1
        for threshold in thresholds:
            tot_pos = 0
            tot_neg = 0
            true_pos = 0
            false_pos = 0
            for filename in tqdm(test_files, disable=True):
                f = h5py.File(filename, "r")
                pred = f.get("exported_data")
                pred = pred[positive_channel, :, :, :]
                mask = pred > threshold

                if object_type == "soma":
                    if filename.split("/")[-1] in doubles:
                        newpos = 2
                    else:
                        newpos = 1

                    labels = measure.label(mask)
                    props = measure.regionprops(labels)
                    if "pos" in filename:
                        num_detected = 0
                        tot_pos += newpos
                        for prop in props:
                            if prop["area"] > size_thresh:
                                if num_detected < newpos:
                                    true_pos += 1
                                    num_detected += 1
                                else:
                                    false_pos += 1
                    elif "neg" in filename:
                        tot_neg += 1
                        for prop in props:
                            if prop["area"] > size_thresh:
                                false_pos += 1
                elif object_type == "axon":
                    filename_lab = filename[:-17] + "-image_3channel_Labels.h5"
                    f = h5py.File(filename_lab, "r")
                    gt = f.get("exported_data")
                    gt = gt[0, :, :, :]
                    pos_labels = gt == 2
                    neg_labels = gt == 1

                    tot_pos += np.sum(pos_labels)
                    tot_neg += np.sum(neg_labels)
                    true_pos += np.sum(np.logical_and(mask, pos_labels))
                    false_pos += np.sum(np.logical_and(mask, neg_labels))
                else:
                    raise ValueError(
                        f"object_type must be axon or soma, not {object_type}"
                    )

            brain_ids_data.append(brain_id)
            recall = true_pos / tot_pos
            recalls.append(recall)
            if true_pos + false_pos == 0:
                precision = 0
            else:
                precision = true_pos / (true_pos + false_pos)
            precisions.append(precision)
            if precision == 0 and recall == 0:
                fscore = 0
            else:
                fscore = 2 * precision * recall / (precision + recall)

            if fscore > best_fscore:
                best_fscore = fscore
                best_thresh = threshold
                best_prec = precision
                best_recall = recall

        best_fscores[brain_id] = (best_fscore, best_thresh)
        best_precisions.append(best_prec)
        best_recalls.append(best_recall)

    for i, brain_id in enumerate(brain_ids_data):
        brain_ids_data[i] = (
            brain_id
            + f" - MaxFS: {best_fscores[brain_id][0]:.2f} @thresh: {best_fscores[brain_id][1]}"
        )

    dict = {
        "ID": brain_ids_data,
        "Recall": recalls,
        "Precision": precisions,
    }
    df = pd.DataFrame(dict)

    print("If this performance is not adequate, improve model and try again")

    if show_plot:
        sns.set(font_scale=2)
        plt.figure(figsize=(8, 8))
        sns.lineplot(
            data=df, x="Recall", y="Precision", hue="ID", estimator=np.amax, ci=False
        )
        plt.scatter(
            best_recall,
            best_prec,
        )
        plt.xlim([0, 1.1])
        plt.ylim([0, 1.1])
        plt.title(f"Brain {brain_id} Validation: {tot_pos}+ {tot_neg}-")
        plt.legend()
        plt.show()

    return best_fscore, best_thresh


def examine_threshold(
    data_dir: str,
    brain_id: str,
    threshold: float,
    object_type: str,
    positive_channel: int,
    doubles: list = [],
    show_plot: bool = True,
):
    """Display results in napari of all subvolumes that were below some performance threshold, at a given threshold.

    Args:
        data_dir (str): Path to directory where brain subvolumes are stored.
        brain_id (str): Brain ID to examine (from _data.py file).
        threshold (float): Threshold to examine.
        object_type (str): soma or axon, the data type being examined.
        positive_channel (int): 0 or 1, Channel that represents neuron in the predictions.
        doubles (list, optional): Filenames of soma subvolumes that contain two somas, if applicable. Defaults to [].
        show_plot (bool, optional): Whether to run napari, useful for pytests when figures should not be displayed. Defaults to True.

    Raises:
        ValueError: If object_type is neither axon nor soma
        ValueError: If positive_channel is not 0 or 1.
    """
    base_dir = data_dir + f"/brain{brain_id}/val/"

    data_files = _find_sample_names(base_dir, dset="", add_dir=True)
    test_files = [file.split(".")[0] + "_Probabilities.h5" for file in data_files]

    size_thresh = 500

    for im_fname, filename in tqdm(
        zip(data_files, test_files), disable=True, total=len(data_files)
    ):
        print(f"*************File: {im_fname}*********")
        f = h5py.File(filename, "r")
        pred = f.get("exported_data")
        pred = pred[positive_channel, :, :, :]
        mask = pred > threshold

        if object_type == "soma":
            if filename.split("/")[-1] in doubles:
                newpos = 2
            else:
                newpos = 1

            labels = measure.label(mask)
            props = measure.regionprops(labels)
            if "pos" in filename:
                num_detected = 0
                for prop in props:
                    area = prop["area"]
                    if area > size_thresh:
                        num_detected += 1
                        print(f"area of detected object: {area}")
                        if num_detected > newpos and show_plot:
                            print(f"Soma false positive Area: {area}")
                            f = h5py.File(im_fname, "r")
                            im = f.get("image_3channel")
                            viewer = napari.Viewer(ndisplay=3)
                            viewer.add_image(im[0, :, :, :], name=filename)
                            viewer.add_image(im[1, :, :, :], name="bg")
                            viewer.add_image(im[2, :, :, :], name="endo")
                            viewer.add_labels(mask)
                            viewer.add_labels(
                                labels == prop["label"],
                                name=f"soma false positive area: {area}",
                            )

                if num_detected == 0 and show_plot:
                    print(f"Soma false negative")
                    f = h5py.File(im_fname, "r")
                    im = f.get("image_3channel")
                    viewer = napari.Viewer(ndisplay=3)
                    viewer.add_image(im[0, :, :, :], name=filename)
                    viewer.add_image(im[1, :, :, :], name="bg")
                    viewer.add_image(im[2, :, :, :], name="endo")
                    viewer.add_labels(mask, name="Soma false negative")

            elif "neg" in filename:
                for prop in props:
                    area = prop["area"]
                    if area > size_thresh and show_plot:
                        print(f"Nonsoma false positive Area: {area}")
                        f = h5py.File(im_fname, "r")
                        im = f.get("image_3channel")
                        viewer = napari.Viewer(ndisplay=3)
                        viewer.add_image(im[0, :, :, :], name=filename)
                        viewer.add_image(im[1, :, :, :], name="bg")
                        viewer.add_image(im[2, :, :, :], name="endo")
                        viewer.add_labels(mask)
                        viewer.add_labels(
                            labels == prop["label"],
                            name=f"nonsoma false positive area: {area}",
                        )
        elif object_type == "axon":
            fname_lab = im_fname.split(".")[0] + "-image_3channel_Labels.h5"
            f = h5py.File(fname_lab, "r")
            gt = f.get("exported_data")
            gt = gt[0, :, :, :]
            if positive_channel == 1:
                pos_labels = gt == 2
                neg_labels = gt == 1
            elif positive_channel == 0:
                pos_labels = gt == 1
                neg_labels = gt == 2
            else:
                raise ValueError(
                    f"positive_channel expected to be 0 or 1 not {positive_channel}"
                )

            true_pos = np.sum(np.logical_and(mask, pos_labels))
            false_pos = np.sum(np.logical_and(mask, neg_labels))
            true_labels = np.sum(pos_labels)

            if true_labels == 0:
                recall = 1
            else:
                recall = true_pos / true_labels

            if true_pos + false_pos == 0:
                precision = 1
            else:
                precision = true_pos / (true_pos + false_pos)

            if (precision < 0.8 or recall) < 0.8 and show_plot:
                f = h5py.File(im_fname, "r")
                im = f.get("image_3channel")
                print(f"prec{precision} recall: {recall}")
                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(im[0, :, :, :], name=f"{im_fname}")
                viewer.add_image(im[1, :, :, :], name="bg")
                viewer.add_image(im[2, :, :, :], name="endo")
                viewer.add_labels(mask, name="mask")
                viewer.add_labels(pos_labels + 2 * neg_labels, name="pos labels")

        else:
            raise ValueError(f"object_type must be axon or soma, not {object_type}")


class ApplyIlastik_LargeImage:
    """Apply ilastik to large image, where chunking is necessary.

    Arguments:
        ilastk_path (str): Path to ilastik executable.
        ilastik_project (str): Path to ilastik project.
        ncpu (int): Number of cpus to use for applying ilastik in parallel.
        object_type (str): Soma for soma detection or axon for axon segmentaiton.
        results_dir: (str or Path): For soma detection, the directory to write detection results.

    Attributes:
        ilastk_path (str): Path to ilastik executable.
        ilastik_project (str): Path to ilastik project.
        ncpu (int): Number of cpus to use for applying ilastik in parallel.
        object_type (str): Soma for soma detection or axon for axon segmentaiton.
        results_dir: (Path): For soma detection, the directory to write detection results.

    """

    def __init__(
        self,
        ilastik_path: str,
        ilastik_project: str,
        ncpu: int,
        data_file: str,
        results_dir: Union[str, Path] = None,
    ):
        with open(data_file) as f:
            data = json.load(f)
        object_type = data["object_type"]
        self.brain2paths = data["brain2paths"]

        self.ilastik_path = ilastik_path
        self.ilastik_project = ilastik_project
        self.ncpu = ncpu
        self.object_type = object_type

        if object_type == "axon":
            if results_dir != None:
                raise ValueError(
                    f"cannot give results_dir for object type {object_type}"
                )
        elif object_type == "soma":
            if isinstance(results_dir, str):
                results_dir = Path(results_dir)
        else:
            raise ValueError(f"object_type must be soma or axon not {object_type}")
        self.results_dir = results_dir

    def apply_ilastik_parallel(
        self,
        brain_id: str,
        layer_names: list,
        threshold: float,
        data_dir: str,
        chunk_size: list,
        max_coords: list = [-1, -1, -1],
        min_coords: list = [-1, -1, -1],
    ):
        """Apply ilastik to large brain, in parallel.

        Args:
            brain_id (str): Brain ID (key in brain2paths in _data.py file).
            layer_names (list): Precomputed layer names to be appended to the base path.
            threshold (float): Threshold for the ilastik predictor.
            data_dir (str or Path): Path to directory where downloaded data will be temporarily stored.
            chunk_size (list): Size of chunks to be used for parallel application of ilastik.
            max_coords (list, optional): Upper bound of bounding box on which to apply ilastk (i.e. does not apply ilastik beyond these bounds). Defaults to [-1, -1, -1].
            min_coords (list, optional): Lower bound of bounding box on which to apply ilastk (i.e. does not apply ilastik beyond these bounds). Defaults to [-1, -1, -1].
        """
        results_dir = self.results_dir
        volume_base_dir = self.brain2paths[brain_id]["base"]
        sample_path = volume_base_dir + layer_names[0]
        vol = CloudVolume(sample_path, parallel=True, mip=0, fill_missing=True)
        shape = vol.shape
        print(f"Processing: {sample_path} with shape {shape} at threshold {threshold}")

        isExist = os.path.exists(data_dir)
        if not isExist:
            print(f"Creating directory: {data_dir}")
            os.makedirs(data_dir)
        else:
            print(f"Downloaded data will be stored in {data_dir}")
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        elif not isinstance(data_dir, Path):
            raise ValueError(f"data_dir must be str or Path")

        if self.object_type == "soma":
            isExist = os.path.exists(results_dir)
            if not isExist:
                print(f"Creating directory: {results_dir}")
                os.makedirs(results_dir)
            else:
                print(f"Downloaded data will be stored in {results_dir}")
        elif self.object_type == "axon":
            mask_dir = volume_base_dir + "axon_mask"
            try:
                CloudVolume(mask_dir)
            except:
                self._make_mask_info(mask_dir, vol)

        corners = _get_corners(
            shape, chunk_size, max_coords=max_coords, min_coords=min_coords
        )
        corners_chunks = [corners[i : i + 100] for i in range(0, len(corners), 100)]

        for corners_chunk in tqdm(corners_chunks, desc="corner chunks"):
            Parallel(n_jobs=self.ncpu)(
                delayed(self._process_chunk)(
                    corner[0],
                    corner[1],
                    volume_base_dir,
                    layer_names,
                    threshold,
                    data_dir,
                    self.object_type,
                    results_dir,
                )
                for corner in tqdm(corners_chunk, leave=False)
            )
            for f in os.listdir(data_dir):
                os.remove(os.path.join(data_dir, f))

    def _make_mask_info(self, mask_dir: str, vol: CloudVolume):
        info = CloudVolume.create_new_info(
            num_channels=1,
            layer_type="segmentation",
            data_type="uint64",  # Channel images might be 'uint8'
            encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
            resolution=vol.resolution,  # Voxel scaling, units are in nanometers
            voxel_offset=vol.voxel_offset,  # x,y,z offset in voxels from the origin
            # mesh            = 'mesh',
            # Pick a convenient size for your underlying chunk representation
            # Powers of two are recommended, doesn't need to cover image exactly
            chunk_size=(128, 128, 2),  # units are voxels
            volume_size=vol.volume_size,  # e.g. a cubic millimeter dataset
        )
        vol_mask = CloudVolume(mask_dir, info=info, compress=False)
        vol_mask.commit_info()

    def _process_chunk(
        self,
        c1: list,
        c2: list,
        volume_base_dir: str,
        layer_names: list,
        threshold: float,
        data_dir: Path,
        object_type: str,
        results_dir: str = None,
    ):
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

        fname = f"image_{c1[0]}_{c1[1]}_{c1[2]}.h5"
        fname = data_dir / fname

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

        fname_prob = str(fname).split(".")[0] + "_Probabilities.h5"
        with h5py.File(fname_prob, "r") as f:
            pred = f.get("exported_data")
            if object_type == "soma":
                fname_results = f"image_{c1[0]}_{c1[1]}_{c1[2]}_somas.txt"
                fname_results = results_dir / fname_results
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
            elif object_type == "axon":
                dir_mask = volume_base_dir + "axon_mask"
                vol_mask = CloudVolume(
                    dir_mask, parallel=1, mip=mip, fill_missing=True, compress=False
                )
                pred = pred[1, :, :, :]
                mask = np.array(pred > threshold).astype("uint64")
                vol_mask[c1[0] : c2[0], c1[1] : c2[1], c1[2] : c2[2]] = mask

    def collect_soma_results(self, brain_id: str):
        """Combine all soma detections and post to neuroglancer. Intended for use after apply_ilastik_parallel.

        Args:
            brain_id (str): ID to process.
        """
        coords = []
        coords_target_space = []
        results_dir = self.results_dir
        onlyfiles = [
            join(results_dir, f)
            for f in listdir(results_dir)
            if isfile(join(results_dir, f))
        ]
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
                        int(round(float(e.strip()) / f))
                        for e, f in zip(coord, div_factor)
                    ]
                    coords.append(coord)
        print(f"{len(coords)} somas detected, first is: {coords_target_space[0]}")
        all_somas_path = results_dir / f"all_somas_{brain_id}.txt"
        print(f"Writing {all_somas_path}...")
        with open(all_somas_path, "w") as f:
            for coord in coords_target_space:
                f.write(f"{coord}")
                f.write("\n")

        if len(coords_target_space) > 10000:
            random.shuffle(coords_target_space)
            point_chunks = [
                coords_target_space[i : i + 10000]
                for i in range(0, len(coords_target_space), 10000)
            ]
            name = "detected_somas_partial"
        else:
            point_chunks = [coords_target_space]
            name = "detected_somas"

        for coords_target_space in point_chunks:
            ng_link = self.brain2paths[brain_id]["val_info"]["url"]
            viz_link = NGLink(ng_link.split("json_url=")[-1])
            ngl_json = viz_link._json

            ngl_json["layers"] = [
                layer for layer in ngl_json["layers"] if layer["type"] != "annotation"
            ]
            ngl_json["layers"].append(
                {"type": "annotation", "points": coords_target_space, "name": name}
            )
            viz_link = create_viz_link_from_json(
                ngl_json, neuroglancer_link="https://viz.neurodata.io/?json_url="
            )
            print(f"Viz link with detections: {viz_link}")

    def collect_axon_results(self, brain_id: str, ng_layer_name: str):
        """Generate neuroglancer link with the axon_mask segmentation. Intended for use after apply_ilastik_parallel

        Args:
            brain_id (str): ID to process.
            ng_layer_name (str): Name of neuroglancer layer in val_info URL with image data.
        """
        ng_link = self.brain2paths[brain_id]["val_info"]["url"]
        viz_link = NGLink(ng_link.split("json_url=")[-1])
        ngl_json = viz_link._json

        ngl_json["layers"] = [
            layer for layer in ngl_json["layers"] if layer["type"] != "annotation"
        ]
        for layer in ngl_json["layers"]:
            if layer["name"] == ng_layer_name:
                source_pieces = layer["source"].split("/")
                source = ""
                for piece in source_pieces[:-1]:
                    source += piece
                    source += "/"
                source += "axon_mask"

                ngl_json["layers"].append(
                    {"type": "segmentation", "source": source, "name": "axon_mask"}
                )
                break
        print(ngl_json)
        viz_link = create_viz_link_from_json(
            ngl_json, neuroglancer_link="https://viz.neurodata.io/?json_url="
        )
        print(f"Viz link with segmentation: {viz_link}")
