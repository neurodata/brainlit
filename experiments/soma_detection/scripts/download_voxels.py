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

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")
tmp_npy = os.path.join(data_dir, "tmp.npy")
print(f"Downloading voxels to {data_dir}")

brains = [1, 2]
s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket("open-neurodata")

with open(os.path.join(exp_dir, "secrets", "az-storage.json")) as secret:
    az_secret = json.load(secret)
connect_str = az_secret["AZURE_STORAGE_CONNECTION_STRING"]
container_name = "datasets"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

d = []
for brain in brains:
    brain_name = "brain%d" % brain

    brain_prefix = f"brainlit/{brain_name}"
    segments_prefix = f"brainlit/{brain_name}_segments"
    skeletons_prefix = f"{segments_prefix}/skeletons"

    brain_url = f"s3://open-neurodata/{brain_prefix}"
    segments_url = f"s3://open-neurodata/{segments_prefix}"

    ngl_sess = NeuroglancerSession(mip=0, url=brain_url, url_segments=segments_url)

    for i, seg_obj in enumerate(bucket.objects.filter(Prefix=skeletons_prefix)):
        seg_id = os.path.basename(seg_obj.key)
        if seg_id != "info":
            radius = 150
            print(
                f"Pulling volume around root {seg_id} of radius {radius}...",
                end="",
                flush=True,
            )
            t0 = time.time()
            img, bbox, vox = ngl_sess.pull_voxel(int(seg_id), 0, radius)
            t1 = time.time()
            dt = np.around(t1 - t0, decimals=3)
            print(f"done in {dt}s")

            pts = bbox.to_list()
            size = bbox.size3()

            # find first y_positive neighbor
            pts[1] += size[1]
            pts[4] += size[1]
            neighbor_bbox = Bbox(pts[:3], pts[3:])
            print(f"Pulling neighbor...", end="", flush=True)
            t0 = time.time()
            neigh_img = ngl_sess.pull_bounds_img(neighbor_bbox)
            t1 = time.time()
            dt = np.around(t1 - t0, decimals=3)
            print(f"done in {dt}s")

            print(f"Uploading root volume to AzureBlobStorage...", end="", flush=True)
            t0 = time.time()
            np.save(tmp_npy, img, allow_pickle=True)
            with open(tmp_npy, "rb") as data:
                blob_name = f"soma-detection/{brain_name}_{seg_id}.npy"
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, blob=blob_name
                )
                blob_client.upload_blob(data, overwrite=True)
            d.append(
                {
                    "seg_id": seg_id,
                    "channel": "image",
                    "label": True,
                    "filepath": blob_name,
                    "intensity": np.sum(img.flatten()),
                }
            )
            t1 = time.time()
            dt = np.around(t1 - t0, decimals=3)
            print(f"done in {dt}s")

            print(
                f"Uploading neighbor volume to AzureBlobStorage...", end="", flush=True
            )
            t0 = time.time()
            np.save(tmp_npy, neigh_img, allow_pickle=True)
            with open(tmp_npy, "rb") as data:
                blob_name = f"soma-detection/{brain_name}_{seg_id}_neighbor.npy"
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, blob=blob_name
                )
                blob_client.upload_blob(data, overwrite=True)
            d.append(
                {
                    "seg_id": seg_id,
                    "channel": "image",
                    "label": False,
                    "filepath": blob_name,
                    "intensity": np.sum(neigh_img.flatten()),
                }
            )
            t1 = time.time()
            dt = np.around(t1 - t0, decimals=3)
            print(f"done in {dt}s")

dataset = pd.DataFrame(d)
dataset.to_csv(os.path.join(data_dir, "dataset.csv"))