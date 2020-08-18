import pytest
from brainlit.utils.ngl_pipeline import NeuroglancerSession
from brainlit.preprocessing.features import *
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox
import glob
import os

from pathlib import Path

URL = (Path(__file__).resolve().parents[1] / "data" / "upload").as_uri()
# URL = "s3://mouse-light-viz/precomputed_volumes/brain1"
SIZE = 2
OFF = [15, 15, 15]

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
    with pytest.raises(NotImplementedError):
        nbrhood.NeighborhoodFeatures(url=URL, size=SIZE, offset=OFF, segment_url="asdf")


def test_fit_bad_inputs():
    nbr = nbrhood.NeighborhoodFeatures(
        url=URL, size=SIZE, offset=[15, 15, 15], segment_url=URL + "_segments"
    )
    with pytest.raises(TypeError):
        nbr.fit(seg_ids=SEGLIST, num_verts=5, file_path="demo", batch_size=1000)


##################
### validation ###
##################


def test_segment_url():
    nbr = neighborhood.NeighborhoodFeatures(
        url=URL, size=[1, 1, 1], offset=[15, 15, 15], segment_url=URL + "_segments",
    )
    df_nbr = nbr.fit([2], 5)
    assert df_nbr.shape == (10, 30)  # 5on, 5off for each swc
    assert not df_nbr.empty


def test_neighborhood():
    nbr = neighborhood.NeighborhoodFeatures(
        url=URL, size=[1, 1, 1], offset=[15, 15, 15]
    )
    df_nbr = nbr.fit([2, 7], 5)
    assert df_nbr.shape == (20, 30)  # 5on, 5off for each swc
    assert not df_nbr.empty


def test_linear_features():
    lin = linear_features.LinearFeatures(url=URL, size=[1, 1, 1], offset=[15, 15, 15])
    lin.add_filter("gaussian", sigma=[1, 1, 0.3])
    df_lin = lin.fit([2, 7], 5)
    assert df_lin.shape == (20, 4)
    assert not df_lin.empty

    df_lin = lin.fit([2, 7], 5, include_neighborhood=True)
    assert df_lin.shape == (20, 31)
    assert not df_lin.empty


def test_add_filter():
    lin = linear_features.LinearFeatures(url=URL, size=[1, 1, 1], offset=[15, 15, 15])
    with pytest.raises(ValueError):
        lin.add_filter("asdf")
    lin.add_filter("gaussian gradient", sigma=[1, 1, 0.3])
    lin.add_filter("gaussian laplace", sigma=[1, 1, 0.3])
    lin.add_filter("gabor", sigma=[1, 1, 0.3], phi=[0, 0], frequency=2)
    lin.add_filter("gabor", sigma=[1, 1, 0.3], phi=[0, np.pi / 2], frequency=2)
    df_lin = lin.fit([2, 7], 5)
    assert df_lin.shape == (20, 7)


def test_file_write():
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)

    lin = linear_features.LinearFeatures(url=URL, size=[1, 1, 1], offset=[15, 15, 15])
    lin.add_filter("gaussian", sigma=[1, 1, 0.3])
    lin.fit([2, 7], 5, file_path="test", batch_size=10)

    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)
    assert files == ["test0_10_2_4.feather", "test10_20_7_4.feather"]

    df_lin = lin.fit(
        [2, 7], 5, file_path="test", batch_size=10, start_seg=7, start_vert=0
    )
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)
    assert files == ["test0_10_7_4.feather"]

    df_lin = lin.fit(
        [2, 7], 5, file_path="test", batch_size=100, start_seg=7, start_vert=0
    )
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)
    assert files == ["test0_10_7_4.feather"]


def test_parallel():
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)

    nbr = neighborhood.NeighborhoodFeatures(
        url=URL, size=[1, 1, 1], offset=[15, 15, 15]
    )
    nbr.fit(seg_ids=[2, 7], num_verts=4, file_path="test", batch_size=4, n_jobs=2)
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)
    assert files == [
        "test0_4_2_1.feather",
        "test0_4_7_1.feather",
        "test4_8_2_3.feather",
        "test4_8_7_3.feather",
    ]

    nbr = neighborhood.NeighborhoodFeatures(
        url=URL, size=[1, 1, 1], offset=[15, 15, 15]
    )
    nbr.fit(seg_ids=[2, 7], num_verts=2, file_path="test", batch_size=6, n_jobs=2)
    files = sorted(glob.glob("*.feather"))
    for f in sorted(files):
        os.remove(f)
    assert files == ["test0_4_2_1.feather", "test0_4_7_1.feather"]
