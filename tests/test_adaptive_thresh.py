import pytest
import brainlight
from brainlight.algorithms.generate_fragments import adaptive_thresh
from brainlight.utils.ngl_pipeline import NeuroglancerSession
import SimpleITK as sitk
import scipy.ndimage

def test_get_seed(get_url):
    print(get_url)
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    assert isinstance(seed, tuple)
    assert img.squeeze().ndim == len(seed)


def test_get_img_T1(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
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


def test_thres_from_gmm(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    thres = adaptive_thresh.thres_from_gmm(img)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)
    assert img_T1_255_array.max() >= thres
    assert img_T1_255_array.min() <= thres


def test_fast_marching_seg(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.fast_marching_seg(img, seed)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1
    s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
    _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
    assert num_features == 1


def test_level_set_seg(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.level_set_seg(img, seed)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1
    s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
    _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
    assert num_features == 1


def test_connected_threshold(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.connected_threshold(img, seed)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1
    s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
    _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
    assert num_features == 1


def test_confidence_connected_threshold(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.confidence_connected_threshold(img, seed)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1
    s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
    _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
    assert num_features == 1


def test_neighborhood_connected_threshold(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.neighborhood_connected_threshold(img, seed)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1
    s = scipy.ndimage.morphology.generate_binary_structure(3, 3)
    _, num_features = scipy.ndimage.measurements.label(labels, structure=s)
    assert num_features == 1


def test_otsu(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.otsu(img)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1


def test_gmm_seg(get_url):
    ngl_session = NeuroglancerSession(get_url)
    img, _ = ngl_session.get_img(2, 400, sx=2, sy=2, sz=2)
    seed = adaptive_thresh.get_seed(ngl_session)
    _, img_T1_255 = adaptive_thresh.get_img_T1(img)
    img_T1_255_array = sitk.GetArrayFromImage(img_T1_255)

    labels = adaptive_thresh.gmm_seg(img)
    assert labels.shape == img_T1_255_array.shape
    assert labels.min() == 0
    assert labels.max() == 1
    assert labels[seed] == 1
