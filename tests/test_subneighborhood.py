import pytest
from brainlit.utils.session import NeuroglancerSession
from brainlit.feature_extraction import neighborhood as nbrhood
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume

from pathlib import Path

URL = (Path(__file__).resolve().parents[1] / "data" / "upload").as_uri()
# URL = "s3://mouse-light-viz/precomputed_volumes/brain1"
SIZE = 2
OFF = [15, 15, 15]


@pytest.fixture
def gen_array():
    nbr = nbrhood.NeighborhoodFeatures(
        url=URL, size=SIZE, offset=[15, 15, 15], segment_url=URL + "_segments"
    )
    df_nbr = nbr.fit([2], 5)
    ind = SIZE * 2 + 1
    arr = df_nbr.iloc[2, 3:].values.reshape((ind, ind, ind))
    return arr


##############
### inputs ###
##############


def test_init_bad_inputs():
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(url=0, size=SIZE, offset=OFF)
    with pytest.raises(NotImplementedError):
        nbrhood.NeighborhoodFeatures(url="asdf", size=SIZE, offset=OFF)
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(url=URL, size=0.5, offset=OFF)
    with pytest.raises(ValueError):
        nbrhood.NeighborhoodFeatures(
            url=URL, size=-1, offset=OFF, segment_url=URL + "_segments"
        )
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(
            url=URL, size=SIZE, offset=12, segment_url=URL + "_segments"
        )
    with pytest.raises(TypeError):
        nbrhood.NeighborhoodFeatures(url=URL, size=SIZE, offset=OFF, segment_url=0)
    with pytest.raises(ValueError):
        nbrhood.NeighborhoodFeatures(url=URL, size=SIZE, offset=OFF, segment_url="asdf")


def test_fit_bad_inputs():
    nbr = nbrhood.NeighborhoodFeatures(
        url=URL, size=SIZE, offset=[15, 15, 15], segment_url=URL + "_segments"
    )
    with pytest.raises(TypeError):
        nbr.fit(10, 5)


##################
### validation ###
##################


def test_2d(gen_array):
    arr = gen_array
    a1 = arr[2, :, :].flatten()
    sub_a1 = nbrhood.subsample(a1, (5, 5), (3, 3)).reshape((3, 3))
    assert np.array_equal(sub_a1, arr[2, 1:4, 1:4])


def test_even(gen_array):
    arr = gen_array
    a1 = arr[2, :, :].flatten()
    sub_a1_even = nbrhood.subsample(a1, (5, 5), (4, 4)).reshape((4, 4))
    assert np.array_equal(sub_a1_even, arr[2, 0:4, 0:4])


def test_3d(gen_array):
    arr = gen_array
    a2 = arr.flatten()
    sub_a2 = nbrhood.subsample(a2, (5, 5, 5), (3, 3, 3)).reshape((3, 3, 3))
    assert np.array_equal(sub_a2, arr[1:4, 1:4, 1:4])
