import pytest
import zarr
import numpy as np
from brainlit.utils.write import zarr_to_omezarr
import os


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


def test_writeome(init_3dzarr):
    zarr_path, data_dir = init_3dzarr
    out_path = data_dir / "fg_ome.zarr"

    assert not os.path.exists(out_path)
    zarr_to_omezarr(zarr_path=zarr_path, out_path=out_path)
    assert os.path.exists(out_path)
