from os.path import join
import boto3
import os
from pathlib import Path
from cloudvolume.lib import Bbox
import numpy as np
from tqdm import tqdm
from brainlit.utils.session import NeuroglancerSession
import matplotlib.pyplot as plt
from brainlit.algorithms.detect_somas import find_somas

viz = True
mip = 1

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")

brains = [1]

s3 = boto3.resource("s3")
bucket = s3.Bucket("open-neurodata")


def classify(brain):
    brain_name = f"brain{brain}"

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

            soma_coords = np.load(vol_filepath, allow_pickle=True)

            ngl_sess = NeuroglancerSession(
                mip=mip, url=brain_url, url_segments=segments_url, use_https=False
            )
            res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]

            volume_coords = np.array(
                os.path.basename(vol_object.key).split("_")
            ).astype(float)
            volume_vox_min = np.round(np.divide(volume_coords[:3], res)).astype(int)
            volume_vox_max = np.round(np.divide(volume_coords[3:], res)).astype(int)

            bbox = Bbox(volume_vox_min, volume_vox_max)
            volume = ngl_sess.pull_bounds_img(bbox)

            rel_soma_coords = np.array(
                [
                    np.round(np.divide(c, res)).astype(int) - volume_vox_min
                    for c in soma_coords
                ]
            )

            try:
                label, rel_pred_centroids, out = find_somas(volume, res)
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
                    out_dir_proj = f"{out_dir}_proj"

                total += len(soma_norms)

                _, axes = plt.subplots(1, 2)

                ax = axes[0]
                vol_proj = np.amax(volume, axis=2)
                ax.imshow(vol_proj, cmap="gray", origin="lower")
                for soma in rel_soma_coords:
                    ax.scatter(soma[1], soma[0], c="none", edgecolor="r")
                if label == 1:
                    for pred_soma in rel_pred_centroids:
                        ax.scatter(pred_soma[1], pred_soma[0], c="b", alpha=0.5)

                ax = axes[1]
                mask_proj = np.amax(out, axis=2)
                ax.imshow(mask_proj, cmap="jet", vmin=0, origin="lower")

                plt.savefig(os.path.join(out_dir_proj, f"{vol_id}.png"))
                plt.close()
    print(f"hits: {hit_count}/{total}")
    print(f"miss: {miss_count}/{total}")


classify(1)
