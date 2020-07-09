from tqdm import tqdm
from glob import glob
import argparse
import numpy as np
from cloudvolume import CloudVolume, Skeleton, storage
from pathlib import Path
import tifffile as tf
from joblib import Parallel, delayed, cpu_count
from cloudvolume.lib import mkdir, touch
import os
from brainlit.utils.upload_to_neuroglancer import chunks, create_image_layer, get_data_ranges, upload_chunks, get_volume_info


def validate_upload(vol, ranges, image):
    pulled_img = vol[
        ranges[0][0] : ranges[0][1],
        ranges[1][0] : ranges[1][1],
        ranges[2][0] : ranges[2][1],
    ]
    if np.sum(pulled_img)==0:
        print('all 0')
        return False
    else:
        print('valid')
        return True


def upload_chunk(vol, ranges, image, progress_dir, to_upload):
    """Push tif image as a chunk in CloudVolume object

    Arguments:
        vol {cloudvolume.CloudVolume} -- volume that will contain image data
        ranges {tuple} -- 3 tuple of lists for image stitch min,max bounds
        image {numpy array} -- 3D image array
    """
    if not str(ranges) in to_upload:
        vol[
            ranges[0][0] : ranges[0][1],
            ranges[1][0] : ranges[1][1],
            ranges[2][0] : ranges[2][1],
        ] = image.T
        print("uploaded")
        if validate_upload(vol, ranges, image):
            print("valid")
            touch(os.path.join(progress_dir, str(ranges)))
    else:
        print("passs")


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

    tiffs = Parallel(tiff_jobs, backend="loky", verbose=50)(
        delayed(tf.imread)("/".join(i)) for i in files
    )
    ranges = Parallel(tiff_jobs, backend='threading')(
        delayed(get_data_ranges)(i, chunk_size) for i in bin_paths
    )
    print("loaded tiffs and bin paths")

    progress_dir = mkdir('./progress/'+str(curr_idx)) # unlike os.mkdir doesn't crash on prexisting
    done_files = set([ z for z in os.listdir(progress_dir) ])
    all_files = set([str(range) for range in ranges])

    to_upload = [ z for z in list(all_files.difference(done_files)) ]
    vol_ = CloudVolume(vol.layer_cloudpath, parallel=False, mip=vol.mip)

    Parallel(tiff_jobs, backend='threading', verbose=50)(
        delayed(upload_chunk)(vol_, r, i, progress_dir, to_upload) for r, i in zip(ranges, tiffs)
    )


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
    mkdir('./progress/')
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
