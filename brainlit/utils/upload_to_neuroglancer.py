from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from cloudvolume import CloudVolume, Skeleton, storage
from pathlib import Path
import tifffile as tf
import contextlib
import joblib
from joblib import Parallel, delayed, cpu_count

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close() 

# chunk data for parallel work
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def create_image_layer(s3_bucket, tif_dimensions, voxel_size, num_resolutions):
    """Creates segmentation layer for skeletons

    Arguments:
        s3_bucket {str} -- path to SWC file
        voxel_size {list} -- 3 floats for voxel size in nm
        num_resolutions {int} -- number of resolutions for the image
    Returns:
        vols {list} -- List of num_resolutions CloudVolume objects, starting from lowest resolution
    """
    # create cloudvolume info
    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="image",
        data_type="uint16",  # Channel images might be 'uint8'
        encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution=voxel_size,  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=[int(d / 4) for d in tif_dimensions],  # units are voxels
        # USING MAXIMUM VOLUME size
        volume_size=[i * 2 ** (num_resolutions - 1) for i in tif_dimensions],
    )
    # get cloudvolume info
    vol = CloudVolume(s3_bucket, info=info, parallel=False)  # compress = False
    # scales resolution up, volume size down
    [
        vol.add_scale((2 ** i, 2 ** i, 2 ** i)) for i in range(num_resolutions)
    ]  # ignore chunk size
    vol.commit_info()
    vols = [
        CloudVolume(s3_bucket, mip=i, parallel=False)  # parallel False, compress
        for i in range(num_resolutions - 1, -1, -1)
    ]
    return vols


def get_data_ranges(bin_path, chunk_size):
    """Get ranges (x,y,z) for chunks to be stitched together in volume

    Arguments:
        bin_path {list} -- binary paths to tif files
        chunk_size {list} -- 3 ints for original tif image dimensions
    Returns:
        x_range {list} -- x-coord int bounds for volume stitch
        y_range {list} -- y-coord int bounds for volume stitch
        z_range {list} -- z-coord int bounds for volume stitch
    """
    x_curr, y_curr, z_curr = 0, 0, 0
    tree_level = len(bin_path)
    for idx, i in enumerate(bin_path):
        scale_factor = 2 ** (tree_level - idx - 1)
        x_curr += int(i[2]) * chunk_size[0] * scale_factor
        y_curr += int(i[1]) * chunk_size[1] * scale_factor
        # flip z axis so chunks go anterior to posterior
        z_curr += int(i[0]) * chunk_size[2] * scale_factor
    x_range = [x_curr, x_curr + chunk_size[0]]
    y_range = [y_curr, y_curr + chunk_size[1]]
    z_range = [z_curr, z_curr + chunk_size[2]]
    return x_range, y_range, z_range


def upload_chunk(vol, ranges, image):
    """Push tif image as a chunk in CloudVolume object

    Arguments:
        vol {cloudvolume.CloudVolume} -- volume that will contain image data
        ranges {tuple} -- 3 tuple of lists for image stitch min,max bounds
        image {numpy array} -- 3D image array
    """
    vol[
        ranges[0][0] : ranges[0][1],
        ranges[1][0] : ranges[1][1],
        ranges[2][0] : ranges[2][1],
    ] = image.T


def parallel_upload_chunks(vol, files, bin_paths, chunk_size, num_workers):
    """Push tif images as chunks in CloudVolume object in Parallel

    Arguments:
        vol {cloudvolume.CloudVolume} -- volume that will contain image data
        files {list} -- strings of tif image filepaths
        bin_paths {list} -- binary paths to tif files
        chunk_size {list} -- 3 ints for original tif image dimensions
        num_workers {int} -- max number of concurrently running jobs
    """
    tiff_jobs = int(num_workers / 2) if num_workers == cpu_count() else num_workers

    with tqdm_joblib(tqdm(desc="Load tiffs", total=len(files))) as progress_bar:
        tiffs = Parallel(tiff_jobs, timeout=1800, backend="multiprocessing", verbose=50)(
            delayed(tf.imread)("/".join(i)) for i in files
        )
    with tqdm_joblib(tqdm(desc="Load ranges", total=len(bin_paths))) as progress_bar:
        ranges = Parallel(tiff_jobs, timeout=1800, backend="multiprocessing", verbose=50)(
            delayed(get_data_ranges)(i, chunk_size) for i in bin_paths
        )
    print("loaded tiffs and bin paths")
    vol_ = CloudVolume(vol.layer_cloudpath, parallel=False, mip=vol.mip)

    with tqdm_joblib(tqdm(desc="Upload chunks", total=len(ranges))) as progress_bar:
        Parallel(tiff_jobs, timeout=1800, backend="multiprocessing", verbose=50)(
            delayed(upload_chunk)(vol_, r, i) for r, i in zip(ranges, tiffs)
        )


def upload_chunks(vol, files, bin_paths, parallel=True):
    """Push tif images into vols with or without joblib Parallel

    Arguments:
        vol {cloudvolume.CloudVolume} -- volume that will contain image data
        files {list} -- strings of tif image filepaths
        bin_paths {list} -- binary paths to tif files
        parallel {bool} -- True to use parallel version, false otherwise
    """
    # all tifs will be this size, should be 528x400x208 for mouselight
    chunk_size = vol.info["scales"][-1]["size"]
    num_workers = len(files) if len(files) < cpu_count() else cpu_count()
    if parallel:
        print("Doing parallel stuff")
        for f, bin_path in tqdm(
            zip(chunks(files, num_workers), chunks(bin_paths, num_workers)),
            total=len(files) // num_workers,
            desc="uploading tiffs",
        ):
            parallel_upload_chunks(vol, f, bin_path, chunk_size, num_workers)
    else:
        print("Not paralleling")
        for f, bin_path in zip(files, bin_paths):
            if vol.mip == len(vol.info["scales"]) - 1:
                img = np.squeeze(tf.imread("/".join(f)))
                vol[:, :, :] = img.T
            else:
                ranges = get_data_ranges(bin_path, chunk_size)
                img = np.squeeze(tf.imread("/".join(f)))
                upload_chunk(vol, ranges, img)


def get_volume_info(image_dir, num_resolutions, channel=0):
    """Get filepaths along the octree-format image directory

    Arguments:
        image_dir {str} -- filepath to HIGHEST LEVEL(lowest res) of octree dir
        num_resolutions {int} -- Number of resolutions for which downsampling has been done
        channel {int} -- Channel number to upload
    Returns:
        files_ordered {list} -- list of file paths, 1st dim contains list for each res
        paths_bin {list} -- list of binary paths, 1st dim contains lists for each res
        vox_size {list} -- list of highest resolution voxel sizes (nm)
        tiff_dims {3-tuple} -- (x,y,z) voxel dimensions for a single tiff image
    """
    files = [str(i).split("/") for i in Path(image_dir).rglob(f"*.{channel}.tif")]
    parent_dirs = len(image_dir.split("/"))

    files_ordered = [
        [i for i in files if len(i) == j + parent_dirs] for j in range(num_resolutions)
    ]
    paths_bin = [
        [[f"{int(j)-1:03b}" for j in k if len(j) == 1] for k in i]
        for i in files_ordered
    ]
    print(f"got files and binary representations of paths.")
    tiff_dims = np.squeeze(tf.imread(image_dir + "/default.0.tif")).T.shape
    transform = open(image_dir + "/transform.txt", "r")
    vox_size = [
        float(s[4:].rstrip("\n")) * (0.5 ** (num_resolutions - 1))
        for s in transform.readlines()
        if "s" in s
    ]
    print(f"got dimensions of volume")
    return files_ordered, paths_bin, vox_size, tiff_dims


def main():
    """
    Runs the script to upload big brain files organized as octree (see https://github.com/neurodata/mouselight_code/issues/1)
    to S3 in neuroglancer format.

    Example:
    >> python upload_to_neuroglancer.py s3://mouse-light-viz/precomputed_volumes/brain1 /cis/local/jacs/data/jacsstorage/samples/2018-08-01/
    """
    parser = argparse.ArgumentParser(
        "Convert a folder of SWC files to neuroglancer format and upload them to the given S3 bucket location."
    )
    parser.add_argument(
        "s3_bucket",
        help="S3 bucket path of the form s3://<bucket-name>/<path-to-layer>",
    )
    parser.add_argument(
        "image_dir",
        help="Path to local directory where image hierarchy lives. Assuming it is formatted as a resolution octree.",
    )
    parser.add_argument(
        "--chosen_res",
        help="Specified resolution to upload. 0 is highest. Default uploads all",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--channel", help="Channel number to upload. Default is 0", default=0, type=int
    )
    parser.add_argument(
        "--num_resolutions",
        help="Number of resoltions for  which downsampling has been done. Default: 7",
        default=7,
        type=int,
    )
    args = parser.parse_args()

    files_ordered, bin_paths, vox_size, tiff_dims = get_volume_info(
        args.image_dir, args.num_resolutions, args.channel
    )
    vols = create_image_layer(args.s3_bucket, tiff_dims, vox_size, args.num_resolutions)
    pbar = tqdm(enumerate(zip(files_ordered, bin_paths)), total=len(files_ordered))
    for idx, item in pbar:
        if args.chosen_res == -1:
            pbar.set_description_str(
                f"uploading chunks to resolution {args.num_resolutions - idx - 1}..."
            )
            upload_chunks(vols[idx], item[0], item[1], parallel=True)
        else:
            if idx == (args.num_resolutions - args.chosen_res - 1):
                pbar.set_description_str(
                    f"uploading chunks to resolution {args.num_resolutions - idx - 1}..."
                )
                upload_chunks(vols[idx], item[0], item[1], parallel=True)


if __name__ == "__main__":
    main()
