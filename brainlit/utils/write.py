import aicspylibczi
import numpy as np
import zarr
from tqdm import tqdm
import dask.array as da
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url
from typing import List
from pathlib import Path

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
        slice, _ = czi.read_image(C=0, Z=0)
        slice = np.squeeze(slice)
    return slice

def czi_to_zarr(czi_path: str, out_dir: str) -> List[str]:
    """Convert  4D czi image to a zarr file(s) at a given directory. Single channel image will produce a single zarr, two channels will produce two.

    Args:
        czi_path (str): Path to czi image.
        out_dir (str): Path to directory where zarr(s) will be written.

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
    zarr_fg = zarr.open(
        fg_path, mode="w", shape=sz, chunks=(200, 200, 10), dtype="uint16"
    )

    for z in tqdm(np.arange(Z), desc="Saving slices foreground..."):
        zarr_fg[:, :, z] = _read_czi_slice(czi, C=0, Z=z)

    if C > 1:  # there is a second (assumed background) channel
        bg_path = out_dir + "bg.zarr"
        zarr_paths.append(bg_path)
        zarr_bg = zarr.open(
            bg_path, mode="w", shape=sz, chunks=(200, 200, 10), dtype="uint16"
        )
        for z in tqdm(np.arange(Z), desc="Saving slices background..."):
            zarr_bg[:, :, z] = _read_czi_slice(czi, C=1, Z=z)
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
