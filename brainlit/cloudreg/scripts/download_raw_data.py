# local imports
from .util import (
    tqdm_joblib,
    chunks,
    S3Url,
)

import time
import os
from io import BytesIO
import argparse
import boto3
from botocore.client import Config
import numpy as np
from PIL import Image
from tqdm import tqdm
from joblib import Parallel, delayed, cpu_count
import math
import tifffile as tf


config = Config(connect_timeout=5, retries={"max_attempts": 5})


def get_out_path(in_path, outdir):
    """Get output path for given tile, maintaining folder structure for Terastitcher

    Args:
        in_path (str): S3 key to raw tile
        outdir (str): Path to local directory to store raw data

    Returns:
        str: Path to store raw tile at.
    """
    head, fname = os.path.split(in_path)
    head_tmp = head.split("/")
    head = f"{outdir}/" + "/".join(head_tmp[-1:])
    idx = fname.find(".")
    fname_new = fname[:idx] + "_corrected.tiff"
    out_path = f"{head}/{fname_new}"
    os.makedirs(head, exist_ok=True)  # succeeds even if directory exists.
    return out_path


def get_all_s3_objects(s3, **base_kwargs):
    """Get all s3 objects with base_kwargs

    Args:
        s3 (boto3.S3.client): an active S3 Client.

    Yields:
        dict: Response object with keys to objects if there are any.
    """
    continuation_token = None
    while True:
        list_kwargs = dict(MaxKeys=1000, **base_kwargs)
        if continuation_token:
            list_kwargs["ContinuationToken"] = continuation_token
        response = s3.list_objects_v2(**list_kwargs)
        yield from response.get("Contents", [])
        if not response.get("IsTruncated"):  # At the end of the list?
            break
        continuation_token = response.get("NextContinuationToken")


def get_list_of_files_to_process(in_bucket_name, prefix, channel):
    """Get paths of all raw data files for a given channel.

    Args:
        in_bucket_name (str): S3 bucket in which raw data live
        prefix (str): Prefix for the S3 path at which raw data live
        channel (int): Channel number to process

    Returns:
        list of str: List of S3 paths for all raw data files
    """
    session = boto3.Session()
    s3_client = session.client("s3", config=config)
    loc_prefixes = s3_client.list_objects_v2(
        Bucket=in_bucket_name, Prefix=prefix, Delimiter="CHN"
    )["CommonPrefixes"]
    loc_prefixes = [i["Prefix"] + f"0{channel}" for i in loc_prefixes]
    all_files = []
    for i in tqdm(loc_prefixes):
        all_files.extend(
            [
                f["Key"]
                for f in get_all_s3_objects(s3_client, Bucket=in_bucket_name, Prefix=i)
            ]
        )
    return all_files


def download_tile(s3, raw_tile_bucket, raw_tile_path, outdir, bias=None):
    """Download single raw data image file from S3 to local directory

    Args:
        s3 (S3.Resource): A Boto3 S3 resource
        raw_tile_bucket (str): Name of bucket with raw data
        raw_tile_path (str): Path to raw data file in S3 bucket
        outdir (str): Local path to store raw data
        bias (np.ndarray, optional): Bias correction multiplied by image before saving. Must be same size as image Defaults to None.
    """
    out_path = get_out_path(raw_tile_path, outdir)
    raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
    # try this unless you get endpoin None error
    # then wait 30 seconds and retry
    try:
        raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))
    except Exception as e:
        print(f"Encountered {e}. Waiting 10 seconds to retry")
        time.sleep(10)
        s3 = boto3.resource("s3")
        raw_tile_obj = s3.Object(raw_tile_bucket, raw_tile_path)
        raw_tile = np.asarray(Image.open(BytesIO(raw_tile_obj.get()["Body"].read())))

    tf.imwrite(out_path, data=raw_tile.astype("uint16"), compress=3, append=False)


def download_tiles(tiles, raw_tile_bucket, outdir):
    """Download a chunk of tiles from S3 to local storage

    Args:
        tiles (list of str): S3 paths to raw data files to download
        raw_tile_bucket (str): Name of bucket where raw data live
        outdir (str): Local path to store raw data at
    """
    session = boto3.Session()
    s3 = session.resource("s3")

    for tile in tiles:
        download_tile(s3, raw_tile_bucket, tile, outdir)


def download_raw_data(in_bucket_path, channel, outdir):
    """Download COLM raw data from S3 to local storage

    Args:
        in_bucket_path (str): Name of S3 bucket where raw dadta live at
        channel (int): Channel number to process
        outdir (str): Local path to store raw data
    """

    input_s3_url = S3Url(in_bucket_path.strip("/"))
    in_bucket_name = input_s3_url.bucket
    in_path = input_s3_url.key
    total_n_jobs = cpu_count()

    # get list of all tiles to correct for  given channel
    all_files = get_list_of_files_to_process(in_bucket_name, in_path, channel)
    total_files = len(all_files)

    # download all the files as tiff
    files_per_proc = math.ceil(total_files / total_n_jobs) + 1
    work = chunks(all_files, files_per_proc)
    with tqdm_joblib(
        tqdm(desc="Downloading tiles", total=total_n_jobs)
    ) as progress_bar:
        Parallel(n_jobs=total_n_jobs, verbose=10)(
            delayed(download_tiles)(files, in_bucket_name, outdir) for files in work
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_bucket_path",
        help="Full path to S3 bucket where raw tiles live. Should be of the form s3://<bucket-name>/<path-to-VW0-folder>/",
        type=str,
    )
    parser.add_argument(
        "--channel",
        help="Channel number to process. accepted values are 0, 1, or 2",
        type=str,
    )
    parser.add_argument(
        "--outdir",
        help="Path to output directory to store corrected tiles. VW0 directory will  be saved here. Default: ~/",
        default="/home/ubuntu/",
        type=str,
    )

    args = parser.parse_args()

    download_raw_data(args.in_bucket_path, args.channel, args.outdir)
