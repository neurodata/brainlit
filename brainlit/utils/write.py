import aicspylibczi
import numpy as np
import zarr
from tqdm import tqdm
import dask.array as da 
from ome_zarr.writer import write_image
from ome_zarr.io import parse_url

def czi_to_zarr(czi_path: str, out_dir: str):
    """Convert  4D czi image to a zarr file(s) at a given directory. Single channel image will produce a single zarr, two channels will produce two.

    Args:
        czi_path (str): Path to czi image.
        out_dir (str): Path to directory where zarr(s) will be written.
    """
    czi = aicspylibczi.CziFile(czi_path)

    slice1 = np.squeeze(czi.read_mosaic(C=0, Z=0, scale_factor=1))
    C = czi.get_dims_shape()[0]["C"][1]
    H = slice1.shape[0]
    W = slice1.shape[1]
    Z = czi.get_dims_shape()[0]["Z"][1]

    print(f"Writing {C} zarrs of shape {H}x{W}x{Z} from czi with dims {czi.get_dims_shape()}")
    sz = np.array([H, W, Z], dtype='int')

    zarr_fg = zarr.open(out_dir + "fg.zarr", mode='w', shape=sz, chunks=(200, 200, 10), dtype="uint16")

    for z in tqdm(np.arange(Z), desc="Saving slices foreground..."):
        zarr_fg[:, :, z] = np.squeeze(czi.read_mosaic(C=0, Z=z, scale_factor=1))

    if C > 1: #there is a second (assumed background) channel
        zarr_bg = zarr.open(out_dir + "bg.zarr", mode='w', shape=sz, chunks=(200, 200, 10), dtype="uint16")
        for z in tqdm(np.arange(Z), desc="Saving slices background..."):
            zarr_bg[:, :, z] = np.squeeze(czi.read_mosaic(C=1, Z=z, scale_factor=1))


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
    write_image(image = dra, group=root, axes = "xyz")