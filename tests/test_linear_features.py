import pytest
from brainlit.utils.ngl_pipeline import NeuroglancerSession
from brainlit.preprocessing.features import *
import numpy as np
import pandas as pd
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox

URL = "s3://mouse-light-viz/precomputed_volumes/brain1"


def test_neighborhood():
    nbr = neighborhood.NeighborhoodFeatures(
        url=URL, size=[1, 1, 1], offset=[15, 15, 15]
    )
    df_nbr = nbr.fit([2, 7], 5)
    assert df_nbr.shape == (20, 4)  # 5on, 5off for each swc
    assert not df_nbr.empty


def test_linear_features():
    lin = linear_features.LinearFeatures(url=URL, size=[1, 1, 1], offset=[15, 15, 15])
    lin.add_filter("gaussian", sigma=[1, 1, 0.3])
    df_lin = lin.fit([2, 7], 5)
    assert df_lin.shape == (20, 4)
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
    assert "Gaussian Gradient" in df_lin["Features"][0]
    assert "Gaussian Laplacian" in df_lin["Features"][0]
    assert "Gabor" in df_lin["Features"][0]
    assert "Gaussian" in df_lin["Features"][0]
    assert len(df_lin["Features"][0]["Gabor"]) == 2
    assert len(df_lin["Features"][0]["Gaussian"]) == 0
