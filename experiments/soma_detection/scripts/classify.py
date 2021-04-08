from os.path import join
import boto3
import os
from pathlib import Path
from cloudvolume.lib import Bbox
from napari import viewer
import numpy as np
from tqdm import tqdm
from brainlit.utils.session import NeuroglancerSession
import napari
from skimage.io import imsave

from skimage import (
    filters,
    morphology,
)

viz = True
mip = 1

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")

brains = [1]

s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")


def contains_somas(volume):
    out = volume.copy()

    t = filters.threshold_otsu(out)
    out = out > t

    clean_selem = morphology.octahedron(2)
    cclean_selem = morphology.octahedron(1)
    out = morphology.erosion(out, clean_selem)
    out = morphology.erosion(out, cclean_selem)

    out, labels = morphology.label(out, background=0, return_num=True)
    for label in np.arange(1, labels + 1):
        A = np.sum(out.flatten() == label)
        if A < 100:
            out[out == label] = 0

    labels, m = morphology.label(out, background=0, return_num=True)
    rel_centroids = np.zeros((m, 3))
    for i, c in enumerate(range(1, m + 1)):
        ids = np.where(labels == c)
        rel_centroids[i] = np.round([np.mean(u) for u in ids]).astype(int)

    label = 0 if (m == 0 or m >= 10) else 1

    return label, rel_centroids, out


def classify(brain):
    brain_name = "brain%d" % brain

    brain_dir = os.path.join(data_dir, brain_name)
    volumes_dir = os.path.join(brain_dir, "volumes")
    results_dir = os.path.join(brain_dir, "results")
    hits_dir = os.path.join(results_dir, "hit")
    miss_dir = os.path.join(results_dir, "miss")

    brain_prefix = f"brainlit/{brain_name}"
    segments_prefix = f"brainlit/{brain_name}_segments"
    somas_prefix = f"brainlit/{brain_name}_somas"

    brain_url = f"s3://open-neurodata/{brain_prefix}"
    segments_url = f"s3://open-neurodata/{segments_prefix}"

    ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)
    res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]

    failed = []
    hit = []
    hit_count = 0
    miss = []
    miss_count = 0
    total = 0
    for vol_object in tqdm(bucket.objects.filter(Prefix=somas_prefix)):
        vol_key = vol_object.key
        vol_id = os.path.basename(vol_object.key)
        if vol_id != "":
            vol_filepath = os.path.join(volumes_dir, f"{vol_id}.npy")
            bucket.download_file(vol_key, vol_filepath)

            volume_coords = np.array(
                os.path.basename(vol_object.key).split("_")
            ).astype(float)
            volume_vox_min = np.round(np.divide(volume_coords[:3], res)).astype(int)
            volume_vox_max = np.round(np.divide(volume_coords[3:], res)).astype(int)

            soma_coords = np.load(vol_filepath, allow_pickle=True)

            rel_soma_coords = np.array(
                [
                    np.round(np.divide(c, res)).astype(int) - volume_vox_min
                    for c in soma_coords
                ]
            )

            bbox = Bbox(volume_vox_min, volume_vox_max)
            volume = ngl_sess.pull_bounds_img(bbox)

            try:
                label, rel_pred_centroids, out = contains_somas(volume)
            except ValueError:
                print(f"failed {vol_id}")
                failed.append(vol_id)
            else:
                if label == 0:
                    miss.append(vol_id)
                    miss_count += 1
                    out_dir = miss_dir
                else:
                    pred_centroids = np.array(
                        [
                            np.multiply(volume_vox_min + c, res)
                            for c in rel_pred_centroids
                        ]
                    )

                    soma_norms = np.linalg.norm(soma_coords, axis=1)
                    pred_norms = np.linalg.norm(pred_centroids, axis=1)
                    match = np.array(
                        [
                            min([abs(prediction - soma) for prediction in pred_norms])
                            < 10e3
                            for soma in soma_norms
                        ]
                    )

                    if match.any():
                        hit.append(vol_id)
                        hit_count += sum(match)
                        miss_count += len(soma_norms) - sum(match)
                        out_dir = hits_dir
                    else:
                        miss.append(vol_id)
                        miss_count += len(soma_norms)
                        out_dir = miss_dir

                total += len(soma_norms)

                viewer = napari.Viewer(ndisplay=3)
                viewer.add_image(volume)
                viewer.add_points(
                    rel_soma_coords,
                    name="ground truth",
                    size=5,
                    symbol="o",
                    face_color=np.array([1, 0, 0, 0.5]),
                )
                if label == 1:
                    viewer.add_points(
                        rel_pred_centroids,
                        name="detection",
                        size=5,
                        symbol="o",
                        face_color=np.array([0, 0, 1, 0.5]),
                    )

                screenshot_path = os.path.join(out_dir, f"{vol_id}_screenshot.png")
                mask_path = os.path.join(out_dir, f"{vol_id}_mask.npy")

                screenshot = viewer.screenshot()
                imsave(screenshot_path, screenshot)
                np.save(mask_path, out)
                viewer.close()
    print(f"hits: {hit_count}/{total}")
    print(f"miss: {miss_count}/{total}")


with napari.gui_qt():
    for brain in brains:
        classify(brain)
