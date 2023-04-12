import pytest
import zarr
import numpy as np
from brainlit.utils.write import zarr_to_omezarr, czi_to_zarr
import os
import shutil
import zipfile
from pathlib import Path


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


##############
### inputs ###
##############


def test_writeome_baddim(init_3dzarr, init_4dzarr):
    zarr_path, data_dir = init_4dzarr
    out_path = data_dir / "fg_ome.zarr"
    with pytest.raises(ValueError, match=r"Conversion only supported for 3D arrays"):
        zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)

    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"
    zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)

    with pytest.raises(
        ValueError,
        match=f"{out_path} already exists, please delete the existing file or change the name of the ome-zarr to be created.",
    ):
        zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)

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
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"

    assert not os.path.exists(out_path)
    zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)
    assert os.path.exists(out_path)
