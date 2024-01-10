import pytest
import zarr
import numpy as np
from brainlit.utils.write import (
    zarr_to_omezarr,
    zarr_to_omezarr_single,
    czi_to_zarr,
    write_trace_layer,
)
import os
import shutil
import zipfile
from pathlib import Path
from cloudvolume import CloudVolume


@pytest.fixture(scope="session")
def init_4dczi(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    czi_path = Path(__file__).parents[0] / "data" / "mosaic_test.czi"

    return czi_path, data_dir


@pytest.fixture(scope="session")
def init_3dzarr(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    zarr_path = data_dir / "fg.zarr"

    z = zarr.open(
        zarr_path, mode="w", shape=(64, 64, 64), dtype="uint16", chunks=(32, 32, 32)
    )
    z[:, :, :] = np.zeros((64, 64, 64))

    return zarr_path, data_dir


@pytest.fixture(scope="session")
def init_4dzarr(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    zarr_path = data_dir / "fg.zarr"

    z = zarr.open(
        zarr_path,
        mode="w",
        shape=(1, 64, 64, 64),
        dtype="uint16",
        chunks=(1, 32, 32, 32),
    )
    z[:, :, :, :] = np.zeros((1, 64, 64, 64))

    return zarr_path, data_dir


@pytest.fixture(scope="function")
def init_omezarr(init_3dzarr):
    res = [1, 1, 2]  # in nm
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"

    if not os.path.exists(out_path):
        zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path, res=res)
    else:
        print("Relying on existing fg_ome zarr file")

    return data_dir, res


##############
### inputs ###
##############


def test_writeome_baddim(init_3dzarr, init_4dzarr):
    # error for 4d zarrs
    zarr_path, data_dir = init_4dzarr
    out_path = data_dir / "fg_ome.zarr"
    with pytest.raises(ValueError, match=r"Conversion only supported for 3D arrays"):
        zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path, res=[1, 1, 1])

    # error if ome already exists
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"
    zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path, res=[1, 1, 1])

    with pytest.raises(
        ValueError,
        match=f"{out_path} already exists, please delete the existing file or change the name of the ome-zarr to be created.",
    ):
        zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path, res=[1, 1, 1])

    shutil.rmtree(out_path)


def test_writeome_single_baddim(init_3dzarr, init_4dzarr):
    # error for 4d zarrs
    zarr_path, data_dir = init_4dzarr
    out_path = data_dir / "fg_ome.zarr"
    with pytest.raises(ValueError, match=r"Conversion only supported for 3D arrays"):
        zarr_to_omezarr_single(zarr_path=zarr_path, out_path=out_path, res=[1, 1, 1])

    # error if ome already exists
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"
    zarr_to_omezarr_single(zarr_path=zarr_path, out_path=out_path, res=[1, 1, 1])

    with pytest.raises(
        ValueError,
        match=f"{out_path} already exists, please delete the existing file or change the name of the ome-zarr to be created.",
    ):
        zarr_to_omezarr_single(zarr_path=zarr_path, out_path=out_path, res=[1, 1, 1])

    shutil.rmtree(out_path)


def test_writezarr_badpar(init_4dczi):
    czi_path, data_dir = init_4dczi
    with pytest.raises(ValueError, match="parallel must be positive integer, not 1"):
        czi_to_zarr(
            czi_path=czi_path, out_dir=str(data_dir), fg_channel=0, parallel="1"
        )


##################
### validation ###
##################


def test_writezarr(init_4dczi):
    czi_path, data_dir = init_4dczi
    zarr_paths = czi_to_zarr(
        czi_path=czi_path, out_dir=str(data_dir), fg_channel=0, parallel=1
    )

    assert len(zarr_paths) == 1

    z = zarr.open(zarr_paths[0])
    assert z.shape == (1, 624, 1756)
    assert z[0, 10, 10] == 411


def test_writezarr_parallel(init_4dczi):
    czi_path, data_dir = init_4dczi
    zarr_paths = czi_to_zarr(
        czi_path=czi_path, out_dir=str(data_dir), fg_channel=0, parallel=2
    )

    assert len(zarr_paths) == 1

    z = zarr.open(zarr_paths[0])
    assert z.shape == (1, 624, 1756)
    assert z[0, 10, 10] == 411


def test_writeome(init_3dzarr):
    res = [1, 1, 2]  # in nm
    dimension_map = {"x": 0, "y": 1, "z": 2}
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"

    assert not os.path.exists(out_path)
    zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path, res=res)
    assert os.path.exists(out_path)

    # check units are micrometers
    ome_zarr = zarr.open(out_path)
    metadata = ome_zarr.attrs["multiscales"][0]

    dimension_names = []
    for dimension in metadata["axes"]:
        assert dimension["unit"] == "micrometer"
        assert dimension["type"] == "space"
        dimension_names.append(dimension["name"])

    # check resolutions are multiples of 2 scaled in xy
    for resolution in metadata["datasets"]:
        lvl = int(resolution["path"])
        true_res = np.multiply(res, [2**lvl, 2**lvl, 1]) / 1000  # in microns
        true_res = [
            true_res[dimension_map[dimension_name]]
            for dimension_name in dimension_names
        ]
        np.testing.assert_almost_equal(
            true_res, resolution["coordinateTransformations"][0]["scale"], decimal=3
        )


def test_writeome_single(init_3dzarr):
    res = [1, 1, 2]  # in nm
    dimension_map = {"x": 0, "y": 1, "z": 2}
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome_single.zarr"

    assert not os.path.exists(out_path)
    zarr_to_omezarr_single(zarr_path=zarr_path, out_path=out_path, res=res)
    assert os.path.exists(out_path)

    # check units are micrometers
    ome_zarr = zarr.open(out_path)
    metadata = ome_zarr.attrs["multiscales"][0]

    dimension_names = []
    for dimension in metadata["axes"]:
        assert dimension["unit"] == "micrometer"
        assert dimension["type"] == "space"
        dimension_names.append(dimension["name"])

    # check resolutions are multiples of 2 scaled in xy
    for resolution in metadata["datasets"]:
        lvl = int(resolution["path"])
        true_res = np.multiply(res, [2**lvl, 2**lvl, 1]) / 1000  # in microns
        true_res = [
            true_res[dimension_map[dimension_name]]
            for dimension_name in dimension_names
        ]
        np.testing.assert_almost_equal(
            true_res, resolution["coordinateTransformations"][0]["scale"], decimal=3
        )


def test_write_trace_layer(init_omezarr):
    data_dir, res = init_omezarr

    write_trace_layer(parent_dir=data_dir, res=res)
    vol_path = "precomputed://file://" + str(data_dir / "traces")
    vol = CloudVolume(vol_path)
    assert vol.info["skeletons"] == "skeletons"
