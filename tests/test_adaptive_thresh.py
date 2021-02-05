import pytest
from brainlit.algorithms.generate_fragments.adaptive_thresh import (
    get_seed,
    get_img_T1,
    thres_from_gmm,
    fast_marching_seg,
    level_set_seg,
    connected_threshold,
    confidence_connected_threshold,
    neighborhood_connected_threshold,
    otsu,
    gmm_seg,
)
import SimpleITK as sitk
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt

##################
### validation ###
##################


def test_get_seed():

    # define voxel
    voxel = (10.131, 30.6001, 100)
    numpy_seed, sitk_seed = get_seed(voxel)
    assert numpy_seed == (10, 30, 100)
    assert sitk_seed == (100, 30, 10)


def test_get_img_T1():
    img = np.array([[[100, 250], [800, 300]], [[1200, 2000], [3000, 2500]]])
    img_T1, img_T1_255 = get_img_T1(img)
    assert type(img_T1) == sitk.Image
    assert type(img_T1_255) == sitk.Image
    assert img_T1_255.GetPixelIDTypeAsString() == "8-bit unsigned integer"


def test_thres_from_gmm():
    # define two groups of Gaussian distribution points with distinct mean values
    G1 = np.round(np.random.normal(loc=40, scale=10, size=(499,1)))
    G2 = np.round(np.random.normal(loc=220, scale=10, size=(499,1)))
    # the minimum value of the high-mean Gaussian distribution determines the threshold
    thre_predicted = np.nanmin(G2)
    # construct a 3D image with the two groups of points
    img = np.append(np.concatenate((G1,G2)),np.array([[0.],[255.]])).reshape((10,10,10))
    # calculate the threshold with `thres_from_gmm`
    thre = thres_from_gmm(img)
    #print(flat_array)
    assert thre == thre_predicted
