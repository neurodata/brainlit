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
    G1 = np.round(np.random.normal(loc=40, scale=10, size=(499, 1)))
    G2 = np.round(np.random.normal(loc=220, scale=10, size=(499, 1)))
    # the minimum value of the high-mean Gaussian distribution determines the threshold
    thre_predicted = np.nanmin(G2)
    # construct a 3D image with the two groups of points
    img = np.append(np.concatenate((G1, G2)), np.array([[0.0], [255.0]])).reshape(
        (10, 10, 10)
    )
    # calculate the threshold with `thres_from_gmm`
    thre = thres_from_gmm(img)
    assert thre == thre_predicted


# def test_level_set_seg():
#    G1 = np.append(np.round(np.random.normal(loc=40, scale=10, size=(499,1))),np.array([0]))
#    G2 = np.append(np.round(np.random.normal(loc=220, scale=10, size=(499,1))),np.array([255]))
#    img = np.concatenate((G1,G2)).reshape((10,10,10))
#    labels = level_set_seg(img,(1,0,1))


def test_connected_threshold():
    # create an image with 4 layers of gray scales
    G1 = np.full((4, 4), 255)
    G2 = np.full((4, 4), 200)
    G3 = np.full((4, 4), 100)
    G4 = np.full((4, 4), 0)
    img = np.concatenate((G1, G2, G3, G4)).reshape(4, 4, 4)
    # seed at the first layer and give a lower_threshold
    labels = connected_threshold(img, [(0, 0, 0)], lower_threshold=150)
    labels_predicted = np.concatenate(
        ((G1 / G1).astype(int), (G2 / G2).astype(int), G3 - G3, G4 - G4)
    ).reshape(4, 4, 4)
    np.testing.assert_array_equal(labels, labels_predicted)
    # seed at the first layer without giving a lower_threshold
    labels = connected_threshold(img, [(0, 0, 0)])
    labels_predicted = np.concatenate(
        ((G1 / G1).astype(int), (G2 / G2).astype(int), G3 - G3, G4 - G4)
    ).reshape(4, 4, 4)
    np.testing.assert_array_equal(labels, labels_predicted)


def test_confidence_connected_threshold():
    # create a data set featured with Gaussian distribution
    G1 = np.append(
        np.round(np.random.normal(loc=125, scale=25, size=(62498, 1))),
        np.array([0, 255]),
    )
    # the data is distributed in the image by the order of each pixel's intensity
    img = np.sort(G1).reshape(250, 250).astype(int)
    # if we set multiplier to be 2.5, we are expected to connect around 99% of the pixels
    labels = confidence_connected_threshold(
        img, [(124, 127)], multiplier=2.5, num_iter=180
    )
    assert sum(sum(labels.astype(float))) / 62500 > 0.98


def test_otsu():
    G1 = np.append(
        np.round(np.random.normal(loc=40, scale=10, size=(499, 1))), np.array([0])
    )
    G2 = np.append(
        np.round(np.random.normal(loc=220, scale=10, size=(499, 1))), np.array([255])
    )
    img = np.concatenate((G1, G2)).reshape((10, 10, 10))
    # seed inside
    labels = otsu(img, (9, 1, 6))
    labels_predicted = np.concatenate(
        ((G1 - G1).astype(int), (G2 / G2).astype(int))
    ).reshape((10, 10, 10))
    np.testing.assert_array_equal(labels, labels_predicted)
    # seed outside
    labels = otsu(img, (1, 1, 3))
    labels_predicted = np.concatenate(
        ((G1 + (255 - G1)).astype(int), (G2 - G2).astype(int))
    ).reshape((10, 10, 10))
    np.testing.assert_array_equal(labels, labels_predicted)


def test_gmm_seg():
    # define two groups of Gaussian distribution points with distinct mean values
    G1 = np.append(
        np.round(np.random.normal(loc=40, scale=10, size=(499, 1))), np.array([0])
    )
    G2 = np.append(
        np.round(np.random.normal(loc=220, scale=10, size=(499, 1))), np.array([255])
    )
    # construct a 3D image with the two groups of points
    img = np.concatenate((G1, G2)).reshape((10, 10, 10))
    # 1s should be labeled to the positions where G2 population are located
    labels_predicted = np.concatenate(
        ((G1 - G1).astype(int), (G2 / G2).astype(int))
    ).reshape((10, 10, 10))
    # the seed is located at a randomly selected point within G2 distribution
    labels = gmm_seg(img, (9, 9, 6))
    np.testing.assert_array_equal(labels, labels_predicted)
