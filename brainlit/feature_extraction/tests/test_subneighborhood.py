import pytest
from brainlit.feature_extraction import neighborhood as nbrhood
from brainlit.utils.tests.test_upload import (
    create_segmentation_layer,
    create_image_layer,
    volume_info,
    upload_volumes_serial,
    paths,
    upload_segmentation,
)
import numpy as np
from pathlib import Path


@pytest.fixture(scope="session")
def vars_local(upload_volumes_serial, upload_segmentation):
    url = upload_volumes_serial.as_uri()
    url_segments, _ = upload_segmentation
    url_segments = url_segments.as_uri()
    return url, url_segments


SIZE = 1
OFF = [1, 1, 0]


@pytest.fixture
def gen_array(vars_local):
    url, url_seg = vars_local
    nbr = nbrhood.NeighborhoodFeatures(
        url=url, radius=SIZE, offset=OFF, segment_url=url_seg
    )
    df_nbr = nbr.fit([2], 5)
    ind = SIZE * 2 + 1
    arr = df_nbr.iloc[2, 3:].values.reshape((ind, ind, ind))
    return arr


##############
### inputs ###
##############


def test_subneighborhood_bad_inputs(gen_array):
    arr = gen_array
    arr_flat = arr[0, :, :].flatten()
    arr_flat3 = arr.flatten()
    with pytest.raises(TypeError):
        nbrhood.subsample("asdf", (3, 3), (1, 1))
    # 2d
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat, 0, (1, 1))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat, (3,), (1, 1))
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat, (3, 3), ("a", "b"))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat, (3, 3), (1,))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat, (3, 3), (-1, -1))
    # 3d
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat3, 0, (1, 1, 1))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat3, (3,), (1, 1, 1))
    with pytest.raises(TypeError):
        nbrhood.subsample(arr_flat3, (3, 3, 3), 0)
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat3, (3, 3, 3), (1,))
    with pytest.raises(ValueError):
        nbrhood.subsample(arr_flat3, (3, 3, 3), (-1, -1, -1))


##################
### validation ###
##################


def test_2d(gen_array):
    arr = gen_array
    a1 = arr[2, :, :].flatten()
    sub_a1 = nbrhood.subsample(a1, (3, 3), (1, 1)).reshape((1, 1))
    assert np.array_equal(sub_a1, arr[2, 1:2, 1:2])


def test_even(gen_array):
    arr = gen_array
    a1 = arr[2, :, :].flatten()
    sub_a1_even = nbrhood.subsample(a1, (3, 3), (2, 2)).reshape((2, 2))
    assert np.array_equal(sub_a1_even, arr[2, 0:2, 0:2])


def test_3d(gen_array):
    arr = gen_array
    a2 = arr.flatten()
    sub_a2 = nbrhood.subsample(a2, (3, 3, 3), (1, 1, 1)).reshape((1, 1, 1))
    assert np.array_equal(sub_a2, arr[1:2, 1:2, 1:2])
