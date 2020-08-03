import math
from cloudvolume import CloudVolume, storage
import numpy as np
import joblib
from joblib import Parallel, delayed
from glob import glob
import argparse
from psutil import virtual_memory
from tqdm.auto import tqdm
import tifffile as tf
from pathlib import Path
from .swc import swc2skeleton
import time
import contextlib

from concurrent.futures.process import ProcessPoolExecutor

# PIL.Image.MAX_IMAGE_PIXELS = None


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


def create_cloud_volume(
    precomputed_path,
    img_size,
    voxel_size,
    num_resolutions=2,
    chunk_size=None,
    parallel=False,
    layer_type="image",
    dtype=None,
):
    """Create CloudVolume volume object and info file.

    Arguments:
        precomputed_path {str} -- cloudvolume path
        img_size {list} -- x,y,z voxel dimensions of tiff images
        voxel_size {list} -- x,y,z dimensions of highest res voxel size (nm)
        
    Keyword Arguments:
        num_resolutions {int} -- the number of resolutions to upload
        chunk_size {list} -- size of chunks to upload. If None, uses img_size/2.
        parallel {bool} -- whether to upload chunks in parallel
        layer_type {str} -- one of "image" or "segmentation"
        dtype {str} -- one of "uint16" or "uint64". If None, uses default for layer type.
    Returns:
        vol {cloudvolume.CloudVolume} -- volume to upload to
    """
    if chunk_size is None:
        chunk_size = [int(i / 2) for i in img_size]
    if dtype is None:
        if layer_type == "image":
            dtype = "uint16"
        elif layer_type == "segmentation":
            dtype = "uint64"
        else:
            raise ValueError(
                f"layer type is {layer_type}, when it should be image or str"
            )

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type=layer_type,
        data_type=dtype,  # Channel images might be 'uint8'
        encoding="raw",  # raw, jpeg, compressed_segmentation, fpzip, kempressed
        resolution=voxel_size,  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        chunk_size=chunk_size,  # units are voxels
        volume_size=[i * 2 ** (num_resolutions - 1) for i in img_size],
        # volume_size=img_size,  # e.g. a cubic millimeter dataset
        skeletons="skeletons",
    )
    vol = CloudVolume(precomputed_path, info=info, parallel=parallel)
    [
        vol.add_scale((2 ** i, 2 ** i, 2 ** i), chunk_size=chunk_size)
        for i in range(num_resolutions)
    ]
    vol.commit_info()
    if layer_type == "image":
        vols = [
            CloudVolume(precomputed_path, mip=i, parallel=parallel)
            for i in range(num_resolutions - 1, -1, -1)
        ]
    elif layer_type == "segmentation":
        skel_info = {
            "@type": "neuroglancer_skeletons",
            "transform": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
            "vertex_attributes": [
                {"id": "radius", "data_type": "float32", "num_components": 1},
                {"id": "vertex_types", "data_type": "float32", "num_components": 1},
                {"id": "vertex_color", "data_type": "float32", "num_components": 4},
            ],
        }
        with storage.SimpleStorage(vol.cloudpath) as stor:
            stor.put_json(str(Path("skeletons") / "info"), skel_info)
        vols = [vol]
    return vols


def get_volume_info(image_dir, num_resolutions, channel=0, extension="tif"):
    """Get filepaths along the octree-format image directory

    Arguments:
        image_dir {str} -- filepath to HIGHEST LEVEL(lowest res) of octree dir
        num_resolutions {int} -- Number of resolutions for which downsampling has been done
        channel {int} -- Channel number to upload
        extension {str} -- File extension of image files
    Returns:
        files_ordered {list} -- list of file paths, 1st dim contains list for each res
        paths_bin {list} -- list of binary paths, 1st dim contains lists for each res
        vox_size {list} -- list of highest resolution voxel sizes (nm)
        tiff_dims {3-tuple} -- (x,y,z) voxel dimensions for a single tiff image
    """

    def RepresentsInt(s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    p = Path(image_dir)
    files = [i.parts for i in p.rglob(f"*.{channel}.{extension}")]
    parent_dirs = len(p.parts)

    files_ordered = [
        [i for i in files if len(i) == j + parent_dirs + 1]
        for j in range(num_resolutions)
    ]
    paths_bin = [
        [[f"{int(j)-1:03b}" for j in k if len(j) == 1 and RepresentsInt(j)] for k in i]
        for i in files_ordered
    ]
    for i, resolution in enumerate(files_ordered):
        for j, filepath in enumerate(resolution):
            files_ordered[i][j] = str(Path(*filepath))
    print(f"got files and binary representations of paths.")

    img_size = np.squeeze(tf.imread(str(p / "default.0.tif"))).T.shape
    transform = open(str(p / "transform.txt"), "r")
    vox_size = [
        float(s[4:].rstrip("\n")) * (0.5 ** (num_resolutions - 1))
        for s in transform.readlines()
        if "s" in s
    ]
    transform = open(str(p / "transform.txt"), "r")
    origin = [int(o[4:].rstrip("\n")) / 1000 for o in transform.readlines() if "o" in o]
    return files_ordered, paths_bin, vox_size, img_size, origin


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


def process(file_path, bin_path, vol):
    array = tf.imread(file_path).T[..., None]
    ranges = get_data_ranges(bin_path, vol.scales[-1]["size"])
    vol[
        ranges[0][0] : ranges[0][1],
        ranges[1][0] : ranges[1][1],
        ranges[2][0] : ranges[2][1],
    ] = array
    return


def upload_volumes(input_path, precomputed_path, num_mips, parallel=False, chosen=-1):
    (files_ordered, paths_bin, vox_size, img_size, _) = get_volume_info(
        input_path, num_mips,
    )

    vols = create_cloud_volume(
        precomputed_path,
        img_size,
        vox_size,
        num_mips,
        parallel=parallel,
        layer_type="image",
    )

    num_procs = min(
        math.floor(
            virtual_memory().total / (img_size[0] * img_size[1] * img_size[2] * 8)
        ),
        joblib.cpu_count(),
    )
    if chosen == -1:
        for mip, vol in enumerate(vols):
            print(f"Started mip {mip}")
            try:
                with tqdm_joblib(
                    tqdm(
                        desc="Creating precomputed volume",
                        total=len(files_ordered[mip]),
                    )
                ) as progress_bar:
                    Parallel(num_procs, timeout=1800, verbose=10)(
                        delayed(process)(f, b, vols[mip],)
                        for f, b in zip(files_ordered[mip], paths_bin[mip])
                    )
            except Exception as e:
                print(e)
                print("timed out on a slice. moving on to the next step of pipeline")
            print(f"Finished mip {mip}")
            print(time.time())
    else:
        try:
            with tqdm_joblib(
                tqdm(
                    desc="Creating precomputed volume", total=len(files_ordered[chosen])
                )
            ) as progress_bar:
                Parallel(num_procs, timeout=1800, verbose=10)(
                    delayed(process)(f, b, vols[chosen],)
                    for f, b in zip(files_ordered[chosen], paths_bin[chosen])
                )
        except Exception as e:
            print(e)
            print("timed out on a slice. moving on to the next step of pipeline")


def create_skel_segids(swc_dir, origin):
    """ Create skeletons to be uploaded as precomputed format

    Arguments:
        swc_dir {str} -- path to consensus swc files
        origin {list} -- x,y,z coordinate of coordinate frame in space in mircons

    Returns:
        skeletons {list} -- swc skeletons to be pushed to bucket
        segids {list} --  list of ints for each swc's label
    """
    p = Path(swc_dir)
    files = [str(i) for i in p.rglob(f"*.swc")]
    skeletons = []
    segids = []
    for i in tqdm(files, desc="converting swcs to neuroglancer format..."):
        skeletons.append(swc2skeleton(i, origin=origin))
        segids.append(skeletons[-1].id)
    return skeletons, segids


def upload_segments(input_path, precomputed_path, num_mips):
    (_, _, vox_size, img_size, origin) = get_volume_info(input_path, num_mips,)
    vols = create_cloud_volume(
        precomputed_path, img_size, vox_size, num_mips, layer_type="segmentation",
    )
    swc_dir = glob(f"{input_path}/*consensus-swcs")[0]
    segments, segids = create_skel_segids(swc_dir, origin)
    for skel in tqdm(segments, desc="uploading segments to S3.."):
        vols[0].skeleton.upload(skel)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert local volume into precomputed volume on S3."
    )
    parser.add_argument(
        "input_path",
        help="Path to directory containing stitched tiles named sequentially.",
    )
    parser.add_argument(
        "precomputed_path",
        help="Path to location where precomputed volume should be stored. Example: s3://<bucket>/<experiment>/<channel>",
    )
    parser.add_argument(
        "layer_type", help="Layer type to upload. One of ['image', 'segmentation']",
    )
    parser.add_argument(
        "--extension",
        help="Extension of stitched files. default is tif",
        default="tif",
        type=str,
    )
    parser.add_argument(
        "--channel", help="Channel to upload to", default=0, type=int,
    )
    parser.add_argument(
        "--num_resolutions",
        help="Number of resoltions for  which downsampling has been done. Default: 7",
        default=2,
        type=int,
    )
    parser.add_argument(
        "--chosen_res",
        help="Specified resolution to upload. 0 is highest. Default uploads all",
        default=-1,
        type=int,
    )

    args = parser.parse_args()
    start = time.time()
    print(time.time())
    if args.layer_type == "image":
        upload_volumes(
            args.input_path,
            args.precomputed_path,
            args.num_resolutions,
            chosen=args.chosen_res,
        )
    elif args.layer_type == "segmentation":
        upload_segments(
            args.input_path, args.precomputed_path, args.num_resolutions,
        )
    else:
        upload_segments(
            args.input_path, args.precomputed_path + "_segments", args.num_resolutions,
        )
        upload_volumes(
            args.input_path,
            args.precomputed_path,
            args.num_resolutions,
            chosen=args.chosen_res,
        )

    print(time.time())
    print(f"total time taken: {int(time.time()-start)} seconds")
