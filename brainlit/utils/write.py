import aicspylibczi
import numpy as np
import zarr
from tqdm import tqdm
import dask.array as da
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url
from typing import List
from pathlib import Path
from joblib import Parallel, delayed


def _read_czi_slice(czi, C, Z):
    """Reads a slice of a czi object, handling whether the czi is a mosaic or not.

    Args:
        czi (aicspylibczi.CziFile): czi object
        C (int): channel
        Z (int): z slice

    Returns:
        np.array: array of the image data
    """
    if czi.is_mosaic():
        slice = np.squeeze(czi.read_mosaic(C=C, Z=Z, scale_factor=1))
    else:
        slice, _ = czi.read_image(C=C, Z=Z)
        slice = np.squeeze(slice)
    return slice


def _write_zrange_thread(zarr_path, czi_path, channel, zs):
    czi = aicspylibczi.CziFile(czi_path)

    zarr_fg = zarr.open(zarr_path)
    for z in zs:
        zarr_fg[:, :, z] = _read_czi_slice(czi, C=channel, Z=z)


def czi_to_zarr(
    czi_path: str, out_dir: str, fg_channel: int = 0, parallel: int = 1
) -> List[str]:
    """Convert  4D czi image to a zarr file(s) at a given directory. Single channel image will produce a single zarr, two channels will produce two.

    Args:
        czi_path (str): Path to czi image.
        out_dir (str): Path to directory where zarr(s) will be written.
        fg_channel (int): Index of foreground channel.
        parallel (int): Number of cpus to use to write zarr.

    Returns:
        list: paths to zarrs that were written
    """
    zarr_paths = []
    czi = aicspylibczi.CziFile(czi_path)

    slice1 = _read_czi_slice(czi, C=0, Z=0)

    C = czi.get_dims_shape()[0]["C"][1]
    H = slice1.shape[0]
    W = slice1.shape[1]
    Z = czi.get_dims_shape()[0]["Z"][1]

    print(
        f"Writing {C} zarrs of shape {H}x{W}x{Z} from czi with dims {czi.get_dims_shape()}"
    )
    sz = np.array([H, W, Z], dtype="int")

    fg_path = out_dir + "fg.zarr"
    zarr_paths.append(fg_path)
    chunk_z = 10
    zarr_fg = zarr.open(
        fg_path, mode="w", shape=sz, chunks=(200, 200, chunk_z), dtype="uint16"
    )

    if parallel == 1:
        for z in tqdm(np.arange(Z), desc="Saving slices foreground..."):
            zarr_fg[:, :, z] = _read_czi_slice(czi, C=fg_channel, Z=z)
    elif isinstance(parallel, int) and parallel > 1:
        z_blocks = [np.arange(i, i + chunk_z) for i in range(0, sz[2], chunk_z)]
        Parallel(n_jobs=parallel)(
            delayed(_write_zrange_thread)(fg_path, czi_path, 1, zs)
            for zs in tqdm(z_blocks, desc="Saving slices foreground...")
        )
    else:
        raise ValueError(f"parallel must be positive integer, not {parallel}")

    for c in range(C):
        if c == fg_channel:
            continue

        bg_path = out_dir + f"channel_{c}.zarr"
        zarr_paths.append(bg_path)
        zarr_bg = zarr.open(
            bg_path, mode="w", shape=sz, chunks=(200, 200, 10), dtype="uint16"
        )

        if parallel == 1:
            for z in tqdm(np.arange(Z), desc="Saving slices background..."):
                zarr_bg[:, :, z] = _read_czi_slice(czi, C=c, Z=z)
        elif parallel > 1:
            z_blocks = [np.arange(i, i + chunk_z) for i in range(0, sz[2], chunk_z)]
            Parallel(n_jobs=parallel)(
                delayed(_write_zrange_thread)(bg_path, czi_path, c, zs)
                for zs in tqdm(z_blocks, desc="Saving slices background...")
            )
    return zarr_paths


def zarr_to_omezarr(zarr_path: str, out_path: str):
    """Convert 3D zarr to ome-zarr.

    Args:
        zarr_path (str): Path to zarr.
        out_path (str): Path of ome-zarr to be created.
    """
    print(f"Converting {zarr_path} to ome-zarr")

    z = zarr.open(zarr_path)
    if len(z.shape) != 3:
        raise ValueError("Conversion only supported for 3D arrays")

    dra = da.from_zarr(zarr_path)

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=dra, group=root, axes="xyz")
