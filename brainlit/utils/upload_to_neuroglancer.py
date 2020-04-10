from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from cloudvolume import CloudVolume, Skeleton, storage
from pathlib import Path
import tifffile as tf
from joblib import Parallel, delayed, cpu_count

# chunk data for parallel work
def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def create_image_layer(s3_bucket, voxel_size, num_resolutions):
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
        chunk_size=[66, 50, 52],  # units are voxels
        # volume_size     =  [528, 400, 208] # units are voxels
        # USING MAXIMUM VOLUME size
        volume_size=[
            528 * 2 ** (num_resolutions - 1),
            400 * 2 ** (num_resolutions - 1),
            208 * 2 ** (num_resolutions - 1),
        ],
    )
    # get cloudvolume info
    vol = CloudVolume(s3_bucket, info=info, parallel=True)
    # scales resolution up, volume size down
    [vol.add_scale((2 ** i, 2 ** i, 2 ** i)) for i in range(num_resolutions)]
    print(vol.info)
    vol.commit_info()
    vols = [
        CloudVolume(s3_bucket, mip=i, parallel=True)
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
    tiffs = Parallel(num_workers, require="sharedmem")(
        delayed(tf.imread)("/".join(i)) for i in files
    )
    ranges = Parallel(num_workers)(
        delayed(get_data_ranges)(i, chunk_size) for i in bin_paths
    )
    print("loaded tiffs and bin paths")
    vols_ = [CloudVolume(vol.layer_cloudpath, parallel=False, mip=vol.mip)] * len(
        ranges
    )
    print("dim of uploaded lists")
    print(len([r for r in ranges]), len([i for i in tiffs]), len([v for v in vols_]))
    Parallel(num_workers)(
        delayed(upload_chunk)(v, r, i) for v, r, i in zip(vols_, ranges, tiffs)
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


def get_file_paths(image_dir, num_resolutions, channel):
    """Get filepaths along the octree-format image directory

    Arguments:
        image_dir {str} -- filepath to HIGHEST LEVEL(lowest res) of octree dir
        num_resolutions {int} -- Number of resolutions for which downsampling has been done
        channel {int} -- Channel number to upload
    Returns:
        files_ordered {list} -- list of file paths, 1st dim contains list for each res
        paths_bin {list} -- list of binary paths, 1st dim contains lists for each res
    """
    files = [str(i).split("/") for i in Path(image_dir).rglob(f"*.{channel}.tif")]
    parent_dirs = len(image_dir.split("/"))
    # Create list with 3 indices, first is the resolution level, second is the file, third are the elements of the file's path
    files_ordered = [
        [i for i in files if len(i) == j + parent_dirs] for j in range(num_resolutions)
    ]
    paths_bin = [
        [[f"{int(j)-1:03b}" for j in k if len(j) == 1] for k in i]
        for i in files_ordered
    ]
    print(f"got files and binary representations of paths.")
    return files_ordered, paths_bin


def main():
    """
    Runs the script to upload big brain files organized as octree (see https://github.com/neurodata/mouselight_code/issues/1)
    to S3 in neuroglancer format.

    Example:
    >> python upload_to_neuroglancer.py s3://mouse-light-viz/precomputed_volumes/brain1 /cis/local/jacs/data/jacsstorage/samples/2018-08-01/ 1 1 1
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
        "voxel_size",
        help="Voxel size for highest resolution image.",
        nargs=3,
        type=float,
    )
    parser.add_argument(
        "chosen_res",
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
    # files = [
    #     str(i).split("/") for i in Path(args.image_dir).rglob(f"*.{args.channel}.tif")
    # ]
    # parent_dirs = len(args.image_dir.split("/"))
    # # Create list with 3 indices, first is the resolution level, second is the file, third are the elements of the file's path
    # files_ordered = [
    #     [i for i in files if len(i) == j + parent_dirs]
    #     for j in range(args.num_resolutions)
    # ]
    # paths_bin = [
    #     [[f"{int(j)-1:03b}" for j in k if len(j) == 1] for k in i]
    #     for i in files_ordered
    # ]
    # print(f"got files and binary representations of paths.")
    files_ordered, bin_paths = get_file_paths(
        args.image_dir, args.num_resolutions, args.channel
    )

    vols = create_image_layer(args.s3_bucket, args.voxel_size, args.num_resolutions)

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
