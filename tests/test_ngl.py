import pytest
from brainlit.utils.ngl_pipeline import NeuroglancerSession
import SimpleITK as sitk
import scipy.ndimage
import numpy as np
from cloudvolume import CloudVolume
from cloudvolume.lib import Bbox


def test_pull():
    url = "s3://mouse-light-viz/precomputed_volumes/brain1"
    ngl = NeuroglancerSession(url, mip=1)
    img, bounds, voxel = ngl.pull_voxel(2, 300)
    img2 = ngl.pull_bounds_img(bounds)
    assert np.all(np.squeeze(np.array(img)) == np.squeeze(np.array(img2)))


def test_pull_v_list():
    url = "s3://mouse-light-viz/precomputed_volumes/brain1"
    ngl = NeuroglancerSession(url, mip=1)
    img, bounds, voxel = ngl.pull_vertex_list(2, np.arange(10))
    assert True


def test_pull_label():
    def _img_to_labels(img, voxel, low=None, up=255):
        img_T1 = sitk.GetImageFromArray(np.squeeze(img), isVector=False)
        img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)
        seed = (int(voxel[2]), int(voxel[1]), int(voxel[0]))
        seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
        seg.CopyInformation(img_T1)
        seg[seed] = 1
        seg = sitk.BinaryDilate(seg, 1)
        seg_con = sitk.ConnectedThreshold(
            img_T1_255, seedList=[seed], lower=int(np.round(low)), upper=up
        )
        vectorRadius = (1, 1, 1)
        kernel = sitk.sitkBall
        seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)
        labels = sitk.GetArrayFromImage(seg_clean)
        return labels

    url = "s3://mouse-light-viz/precomputed_volumes/brain1"
    ngl = NeuroglancerSession(url, mip=1)
    ngl_seg = NeuroglancerSession(url + "_seg", mip=1)
    img, bounds, voxel = ngl.pull_chunk(2, 300)
    label = _img_to_labels(img, voxel, low=11)
    ngl_seg.push(label, bounds)
    label2 = ngl_seg.pull_bounds_seg(bounds)
    assert np.all(np.squeeze(np.array(label) == np.squeeze(np.array(label2))))
