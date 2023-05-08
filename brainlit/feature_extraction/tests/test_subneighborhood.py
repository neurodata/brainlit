import pytest
from brainlit.feature_extraction import neighborhood as nbrhood
import numpy as np
from pathlib import Path

top_level = Path(__file__).parents[3] / "data"
url = (top_level / "test_upload").as_uri()
url_seg = url + "_segments"
url = url + "/serial"

SIZE = 1
OFF = [1, 1, 0]


@pytest.fixture
def gen_array():
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
