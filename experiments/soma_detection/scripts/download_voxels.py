from brainlit.utils.session import NeuroglancerSession
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
from pathlib import Path
from azure.storage.blob import BlobServiceClient
import numpy as np

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")
tmp_npy = os.path.join(data_dir, "tmp.npy")
print(f"Downloading voxels to {data_dir}")

brains = [1, 2]
s3 = boto3.resource("s3", config=Config(signature_version=UNSIGNED))
bucket = s3.Bucket("open-neurodata")

connect_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
container_name = "datasets"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)

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
            print(seg_id)
            img, bbox, vox = ngl_sess.pull_voxel(int(seg_id), 0, 150)
            np.save(tmp_npy, img, allow_pickle=True)

            with open(tmp_npy, 'rb') as data:
                blob_name = f"soma-detection/{brain_name}_{seg_id}.npy"
                blob_client = blob_service_client.get_blob_client(
                    container=container_name, blob=blob_name
                )
                blob_client.upload_blob(data, overwrite=True)
