import pytest
import zarr
import numpy as np
from brainlit.utils.write import zarr_to_omezarr, czi_to_zarr
import os
import zipfile
from pathlib import Path


@pytest.fixture(scope="session")
def init_4dczi(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("data")
    czi_path = data_dir / "test.czi"
    zip_path = (
        Path(__file__).parents[3] / "experiments" / "sriram" / "data" / "test.czi.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "r",
    ) as zip_ref:
        zip_ref.extractall(data_dir)
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


def test_writeome_baddim(init_4dzarr):
    zarr_path, data_dir = init_4dzarr
    out_path = data_dir / "fg_ome.zarr"
    with pytest.raises(ValueError, match=r"Conversion only supported for 3D arrays"):
        zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)


##################
### validation ###
##################


def test_writezarr(init_4dczi):
    czi_path, data_dir = init_4dczi
    zarr_paths = czi_to_zarr(czi_path=czi_path, out_dir=str(data_dir), fg_channel=1)

    assert len(zarr_paths) == 2

    for zarr_path, value in zip(zarr_paths, [63, 8]):
        z = zarr.open(zarr_path)
        assert z.shape == (1998, 2009, 40)
        assert z[100,100,10] == value



def test_writeome(init_3dzarr):
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"

    assert not os.path.exists(out_path)
    zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)
    assert os.path.exists(out_path)
