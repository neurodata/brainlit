import numpy as np
from brainlit.algorithms.detect_somas import find_somas
import brainlit
from brainlit.utils.session import NeuroglancerSession
from cloudvolume.lib import Bbox
import os
from pytest import raises

##############
### inputs ###
##############


def test_find_somas_bad_input():
    volume = []
    res = ""

    # test input volume is numpy.ndarray
    with raises(TypeError, match=r"should be <class 'numpy.ndarray'>"):
        find_somas(volume, res)
    volume = np.array([1, "a", 2])
    with raises(
        TypeError, match=r"elements should be \(<class 'int'>, <class 'float'>\)."
    ):
        find_somas(volume, res)
    volume = np.zeros((2, 2))
    # test input volume must be three-dimensional
    with raises(ValueError, match=r"Input volume must be three-dimensional"):
        find_somas(volume, res)
    volume = np.zeros((19, 19, 21))
    # test input volume has to be at least 20x20xNz
    with raises(ValueError, match=r"Input volume is too small"):
        find_somas(volume, res)
    volume = np.zeros((50, 50, 50))

    # test res should be list
    with raises(TypeError, match=r"should be <class 'list'>"):
        find_somas(volume, res)
    res = [1, "a", 2]
    with raises(
        TypeError, match=r"elements should be \(<class 'int'>, <class 'float'>\)."
    ):
        find_somas(volume, res)
    res = [100, 100.0]
    with raises(ValueError, match=r"Resolution must be three-dimensional"):
        find_somas(volume, res)
    res = [100, 100.0, 0]
    with raises(ValueError, match=r"Resolution must be non-zero at every position"):
        find_somas(volume, res)


##################
### validation ###
##################


def test_detect_output():
    # download a volume
    dir = "s3://open-neurodata/brainlit/brain1"
    dir_segments = "s3://open-neurodata/brainlit/brain1_segments"
    volume_keys = "4807349.0_3827990.0_2922565.75_4907349.0_3927990.0_3022565.75"
    mip = 3
    ngl_sess = NeuroglancerSession(
        mip=mip, url=dir, url_segments=dir_segments, use_https=False
    )
    res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
    volume_coords = np.array(os.path.basename(volume_keys).split("_")).astype(float)
    volume_vox_min = np.round(np.divide(volume_coords[:3], res)).astype(int)
    volume_vox_max = np.round(np.divide(volume_coords[3:], res)).astype(int)
    bbox = Bbox(volume_vox_min, volume_vox_max)
    img = ngl_sess.pull_bounds_img(bbox)
    # apply soma detector
    label, rel_centroids, out = find_somas(img, res)
    # check output type
    assert type(label) == bool
    assert type(rel_centroids) == np.ndarray
    assert type(out) == np.ndarray
    # check output dimension
    assert rel_centroids.shape[1] == 3
    assert out.shape == (160, 160, 50)
