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
import os
from cloudvolume import CloudVolume
import json


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
        zarr_fg[z, :, :] = _read_czi_slice(czi, C=channel, Z=z)


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

    sz = np.array([Z, H, W], dtype="int")
    chunk_z = 10
    chunk_sz = (chunk_z, 200, 200)
    print(f"Writing {C} zarrs of shape {sz} from czi with dims {czi.get_dims_shape()}")

    fg_path = Path(out_dir) / "fg.zarr"
    zarr_paths.append(fg_path)
    zarr_fg = zarr.open(fg_path, mode="w", shape=sz, chunks=chunk_sz, dtype="uint16")

    if parallel == 1:
        for z in tqdm(np.arange(Z), desc="Saving slices foreground..."):
            zarr_fg[z, :, :] = _read_czi_slice(czi, C=fg_channel, Z=z)
    elif isinstance(parallel, int) and parallel > 1:
        z_blocks = [
            np.arange(i, np.amin([i + chunk_z, sz[0]]))
            for i in range(0, sz[0], chunk_z)
        ]
        Parallel(n_jobs=parallel, backend="threading")(
            delayed(_write_zrange_thread)(fg_path, czi_path, channel=fg_channel, zs=zs)
            for zs in tqdm(z_blocks, desc="Saving slices foreground...")
        )
    else:
        raise ValueError(f"parallel must be positive integer, not {parallel}")

    for c in range(C):
        if c == fg_channel:
            continue

        bg_path = Path(out_dir) / f"channel_{c}.zarr"
        zarr_paths.append(bg_path)
        zarr_bg = zarr.open(
            bg_path, mode="w", shape=sz, chunks=chunk_sz, dtype="uint16"
        )

        if parallel == 1:
            for z in tqdm(np.arange(Z), desc="Saving slices background..."):
                zarr_bg[z, :, :] = _read_czi_slice(czi, C=c, Z=z)
        elif parallel > 1:
            Parallel(n_jobs=parallel, backend="threading")(
                delayed(_write_zrange_thread)(bg_path, czi_path, channel=c, zs=zs)
                for zs in tqdm(z_blocks, desc="Saving slices background...")
            )
    return zarr_paths


def zarr_to_omezarr(zarr_path: str, out_path: str, res: list):
    """Convert 3D zarr to ome-zarr.

    Args:
        zarr_path (str): Path to zarr.
        out_path (str): Path of ome-zarr to be created.
        res (list): List of xyz resolution values in nanometers.

    Raises:
        ValueError: If zarr to be written already exists.
        ValueError: If conversion is not 3D array.
    """
    if os.path.exists(out_path):
        raise ValueError(
            f"{out_path} already exists, please delete the existing file or change the name of the ome-zarr to be created."
        )

    print(f"Converting {zarr_path} to ome-zarr")

    z = zarr.open(zarr_path)
    if len(z.shape) != 3:
        raise ValueError("Conversion only supported for 3D arrays")

    dra = da.from_zarr(zarr_path)

    store = parse_url(out_path, mode="w").store
    root = zarr.group(store=store)
    write_image(image=dra, group=root, axes="zxy")
    _edit_ome_metadata(out_path, res)


def _edit_ome_metadata(out_path: str, res: list):
    res = np.divide([res[-1], res[0], res[1]], 1000)
    ome_zarr = zarr.open(
        out_path,
        "r+",
    )
    metadata_edit = ome_zarr.attrs["multiscales"]
    for i in range(3):
        metadata_edit[0]["axes"][i]["unit"] = "micrometer"
    for i, dataset in enumerate(metadata_edit[0]["datasets"]):
        new_res = list(
            np.multiply(dataset["coordinateTransformations"][0]["scale"], res)
        )
        metadata_edit[0]["datasets"][i]["coordinateTransformations"][0][
            "scale"
        ] = new_res
    ome_zarr.attrs["multiscales"] = metadata_edit


def write_trace_layer(parent_dir: str, res: list):
    """Write precomputed layer (info file) for trace skeletons associated with an ome zarr file.

    Args:
        parent_dir (str): Path to directory which holds fg_ome.zarr and where traces layer will be written.
        res (list): List of xyz resolution values in nanometers.
    """
    if isinstance(parent_dir, str):
        parent_dir = Path(parent_dir)

    traces_dir = parent_dir / "traces"
    z = zarr.open_array(parent_dir / "fg_ome.zarr" / "0")
    volume_size = [z.shape[1], z.shape[2], z.shape[0]]
    chunk_size = [z.chunks[1], z.chunks[2], z.chunks[0]]
    outpath = f"precomputed://file://" + str(traces_dir)

    info = CloudVolume.create_new_info(
        num_channels=1,
        layer_type="segmentation",
        data_type="uint16",
        encoding="raw",
        resolution=res,  # Voxel scaling, units are in nanometers
        voxel_offset=[0, 0, 0],  # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size=chunk_size,  # units are voxels
        volume_size=volume_size,  # e.g. a cubic millimeter dataset
        skeletons="skeletons",
    )
    vol = CloudVolume(outpath, info=info, compress=False)
    vol.commit_info()
    vol.skeleton.meta.commit_info()

    # remove vertex type attribute because it is a uint8 and incompatible with neuroglancer
    info_path = traces_dir / "skeletons/info"
    with open(info_path) as f:
        data = json.load(f)
        for i, attr in enumerate(data["vertex_attributes"]):
            if attr["id"] == "vertex_types":
                data["vertex_attributes"].pop(i)
                break

    with open(info_path, "w") as f:
        json.dump(data, f)
