# local imports
from .util import imgResample, tqdm_joblib, get_bias_field

import argparse
from tqdm import tqdm
import SimpleITK as sitk
import numpy as np
from cloudvolume import CloudVolume
import tinybrain
from joblib import Parallel, delayed, cpu_count
from psutil import virtual_memory
import math


def process_slice(bias_slice, z, data_orig_path, data_bc_path):
    """Correct and upload a single slice of data

    Args:
        bias_slice (sitk.Image): Slice of illumination correction
        z (int): Z slice of data to apply correction to
        data_orig_path (str): S3 path to source data that needs to be corrected
        data_bc_path (str): S3 path where corrected data will be stored
    """
    data_vol = CloudVolume(
        data_orig_path, parallel=False, progress=False, fill_missing=True
    )
    data_vol_bc = CloudVolume(
        data_bc_path, parallel=False, progress=False, fill_missing=True
    )
    data_vols_bc = [
        CloudVolume(data_bc_path, mip=i, parallel=False)
        for i in range(len(data_vol_bc.scales))
    ]
    # convert spcing rom nm to um
    new_spacing = np.array(data_vol.scales[0]["resolution"][:2]) / 1000
    bias_upsampled_sitk = imgResample(
        bias_slice, new_spacing, size=data_vol.scales[0]["size"][:2]
    )
    bias_upsampled = sitk.GetArrayFromImage(bias_upsampled_sitk)
    data_native = np.squeeze(data_vol[:, :, z]).T
    data_corrected = data_native * bias_upsampled
    img_pyramid = tinybrain.downsample_with_averaging(
        data_corrected.T[:, :, None],
        factor=(2, 2, 1),
        num_mips=len(data_vol_bc.scales) - 1,
    )
    data_vol_bc[:, :, z] = data_corrected.T.astype("uint16")[:, :, None]
    for i in range(len(data_vols_bc) - 1):
        data_vols_bc[i + 1][:, :, z] = img_pyramid[i].astype("uint16")


def correct_stitched_data(data_s3_path, out_s3_path, resolution=15, num_procs=12):
    """Correct illumination inhomogeneity in stitched precomputed data on S3 and upload result back to S3 as precomputed

    Args:
        data_s3_path (str): S3 path to precomputed volume that needs to be illumination corrected
        out_s3_path (str): S3 path to store corrected precomputed volume
        resolution (int, optional): Resolution in microns at which illumination correction is computed. Defaults to 15.
        num_procs (int, optional): Number of proceses to use when uploading data to S3. Defaults to 12.
    """
    # create vol
    vol = CloudVolume(data_s3_path)
    mip = 0
    for i in range(len(vol.scales)):
        if vol.scales[i]["resolution"][0] <= resolution * 1000:
            mip = i
    vol_ds = CloudVolume(
        data_s3_path, mip, parallel=False, fill_missing=True, progress=True
    )

    # make sure num procs isn't too large for amount of memory needed
    mem = virtual_memory()
    num_processes = min(
        math.floor(
            mem.total
            / (
                (np.prod(vol.scales[0]["size"][:2]))
                # multiply by bytes per voxel (uint16 = 2 bytes)
                * 2
                # fudge factor
                # need 2 copies of full res image, 1 full res bias, 1 full res corrected image, and image downsampled at 6 resolutions
                * 2 ** 7
            )
        ),
        cpu_count(),
    )
    num_procs = num_processes
    print(f"using {num_procs} processes for bias correction")

    # create new vol if it doesnt exist
    vol_bc = CloudVolume(out_s3_path, info=vol.info.copy())
    vol_bc.commit_info()

    # download image at low res
    data = sitk.GetImageFromArray(np.squeeze(vol_ds[:, :, :]).T)
    data.SetSpacing(np.array(vol_ds.scales[mip]["resolution"]) / 1000)

    bias = get_bias_field(data, scale=0.125)
    bias_slices = [bias[:, :, i] for i in range(bias.GetSize()[-1])]
    try:
        with tqdm_joblib(
            tqdm(desc=f"Uploading bias corrected data...", total=len(bias_slices))
        ) as progress_bar:
            Parallel(num_procs, timeout=3600, verbose=10)(
                delayed(process_slice)(bias_slice, z, data_s3_path, out_s3_path)
                for z, bias_slice in enumerate(bias_slices)
            )
    except:
        print("timed out on bias correcting slice. moving to next step.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Correct whole brain bias field in image at native resolution."
    )
    parser.add_argument(
        "data_s3_path",
        help="full s3 path to data of interest as precomputed volume. must be of the form `s3://bucket-name/path/to/channel`",
    )
    parser.add_argument("out_s3_path", help="S3 path to save output results")
    parser.add_argument(
        "--num_procs", help="number of processes to use", default=15, type=int
    )
    parser.add_argument(
        "--resolution",
        help="max resolution for computing bias correction in microns",
        default=15,
        type=float,
    )
    args = parser.parse_args()

    correct_stitched_data(
        args.data_s3_path, args.out_s3_path, args.resolution, args.num_procs
    )
