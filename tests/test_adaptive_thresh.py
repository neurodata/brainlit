import pytest
import brainlit
from brainlit.algorithms.generate_fragments import adaptive_thresh
from brainlit.utils.ngl_pipeline import NeuroglancerSession
import SimpleITK as sitk
import scipy.ndimage
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox


def test_get_seed():
    ngl_session = NeuroglancerSession()
    img, _, vox = ngl_session.pull_voxel(2, 400, nx=1, ny=1, nz=1)
    seed = adaptive_thresh.get_seed(vox)
    assert isinstance(seed, tuple)
    assert img.squeeze().ndim == len(seed)


def test_get_img_T1():
    ngl_session = NeuroglancerSession()
    img, _, _ = ngl_session.pull_voxel(2, 400, nx=1, ny=1, nz=1)
    img_T1, img_T1_255 = adaptive_thresh.get_img_T1(img)
    correct_shape = img.squeeze().shape
    img_T1_array = sitk.GetArrayFromImage(img_T1)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
    assert isinstance(img_T1, sitk.Image)
    assert isinstance(img_T1_255, sitk.Image)
    assert img_T1_array.shape == correct_shape
    assert img_T1_255_array.shape == correct_shape
    assert img_T1_255_array.max() <= 255
    assert img_T1_255_array.min() >= 0


# def test_thres_from_gmm():
#     ngl_session = NeuroglancerSession()
#     img, _, _ = ngl_session.pull_voxel(2, 400, nx=1, ny=1, nz=1)
#     thres = adaptive_thresh.thres_from_gmm(img)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#     assert img_T1_255_array.max() >= thres
#     assert img_T1_255_array.min() <= thres
#
#
# def test_fast_marching_seg():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400, nx=1, ny=1, nz=1)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.fast_marching_seg(img, seed)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
#     s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
#     _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
#     assert num_features == 1
#
#
# def test_level_set_seg():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400, nx=1, ny=1, nz=1)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.level_set_seg(img, seed)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
#     s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
#     _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
#     assert num_features == 1
#
#
# def test_connected_threshold():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.connected_threshold(img, seed)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
#     s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
#     _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
#     assert num_features == 1
#
#
# def test_confidence_connected_threshold():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.confidence_connected_threshold(img, seed)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
#     s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
#     _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
#     assert num_features == 1
#
#
# def test_neighborhood_connected_threshold():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.neighborhood_connected_threshold(img, seed)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
#     s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
#     _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
#     assert num_features == 1
#
#
# def test_otsu():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.otsu(img)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
#
#
# def test_gmm_seg():
#     ngl_session = NeuroglancerSession()
#     img, _, vox = ngl_session.pull_voxel(2, 400)
#     seed = adaptive_thresh.get_seed(vox)
#     _, img_T1_255 = adaptive_thresh.get_img_T1(img)
#     img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
#
#     labels = adaptive_thresh.gmm_seg(img)
#     assert labels.shape == img_T1_255_array.shape
#     assert labels.min() == 0
#     assert labels.max() == 1
#     assert labels[seed] == 1
