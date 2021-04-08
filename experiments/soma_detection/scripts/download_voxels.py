from cloudvolume.lib import Bbox
from brainlit.utils.session import NeuroglancerSession
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
import numpy as np
import json
import time
import pandas as pd

radius = 150

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")
volumes_dir = os.path.join(data_dir, "volumes", str(radius))
print(f"Downloading voxels to {data_dir}")

brains = [1, 2]
s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket("open-neurodata")

with open(os.path.join(exp_dir, "secrets", "az-storage.json")) as secret:
    az_secret = json.load(secret)
connect_str = az_secret["AZURE_STORAGE_CONNECTION_STRING"]
container_name = "datasets"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

for brain in brains:
    brain_name = "brain%d" % brain

    brain_prefix = f"brainlit/{brain_name}"
    segments_prefix = f"brainlit/{brain_name}_segments"
    skeletons_prefix = f"{segments_prefix}/skeletons"

    brain_url = f"s3://open-neurodata/{brain_prefix}"
    segments_url = f"s3://open-neurodata/{segments_prefix}"

    ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)

    for seg_obj in bucket.objects.filter(Prefix=skeletons_prefix):
        seg_id = os.path.basename(seg_obj.key)
        if seg_id != "info":
            volume_filepath = os.path.join(volumes_dir, brain_name, f"{seg_id}.npy")
            print(
                f"Pulling volume around root {seg_id} of radius {radius}...",
                end="",
                flush=True,
            )
            t0 = time.time()
            img, bbox, vox = ngl_sess.pull_voxel(int(seg_id), 0, radius)
            t1 = time.time()
            dt = np.around(t1 - t0, decimals=3)
            print(f"done in {dt}s {bbox}")
            d = {"bbox": np.array(bbox.to_list()), "volume": img}
            np.save(volume_filepath, d, allow_pickle=True)

            pts = bbox.to_list()
            size = bbox.size3()

            neighbor_dims = [(0, 3), (1, 4), (2, 5)]
            neighbor_dirs = [1, -1]
            # find 6-connected neighbors
            for i, neighbor_dim in enumerate(neighbor_dims):
                m = neighbor_dim[0]
                n = neighbor_dim[1]
                for j, neighbor_dir in enumerate(neighbor_dirs):
                    neigh_pts = pts.copy()
                    neigh_id = i * 2 + j
                    neighvolume_filepath = os.path.join(
                        volumes_dir, brain_name, f"{seg_id}_neigh{neigh_id}.npy"
                    )
                    neigh_pts[m] += neighbor_dir * size[i]
                    neigh_pts[n] += neighbor_dir * size[i]

                    # handle possible negative coordinates
                    neigh_pts = [max(0, c) for c in neigh_pts]

                    neighbor_bbox = Bbox(neigh_pts[:3], neigh_pts[3:])
                    print(
                        f"Pulling neighbor #{neigh_id} {neighbor_bbox}...",
                        end="",
                        flush=True,
                    )
                    t0 = time.time()
                    neigh_img = ngl_sess.pull_bounds_img(neighbor_bbox)
                    t1 = time.time()
                    dt = np.around(t1 - t0, decimals=3)
                    print(f"done in {dt}s")
                    d = {"bbox": neigh_pts, "volume": neigh_img}
                    np.save(neighvolume_filepath, d, allow_pickle=True)
