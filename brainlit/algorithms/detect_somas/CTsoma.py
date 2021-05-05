from typing import Tuple
import numpy as np
from skimage import filters, morphology, measure
import pandas as pd
from scipy import ndimage
from brainlit.utils.util import check_type, check_iterable_type
import os
from brainlit.algorithms.detect_somas import find_somas
from pytest import raises
from pathlib import Path
from brainlit.utils.session import NeuroglancerSession
from cloudvolume.lib import Bbox

# # download a volume
# dir = "s3://open-neurodata/brainlit/brain1"
# dir_segments = "s3://open-neurodata/brainlit/brain1_segments"
# volume_keys = "4807349.0_3827990.0_2922565.75_4907349.0_3927990.0_3022565.75"
# mip = 1
# ngl_sess = NeuroglancerSession(mip=mip, url=dir, url_segments=dir_segments, use_https=False)
# res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
# volume_coords = np.array(os.path.basename(volume_keys).split("_")).astype(float)
# volume_vox_min = np.round(np.divide(volume_coords[:3], res)).astype(int)
# volume_vox_max = np.round(np.divide(volume_coords[3:], res)).astype(int)
# bbox = Bbox(volume_vox_min, volume_vox_max)
# img = ngl_sess.pull_bounds_img(bbox)
# # apply soma detector
# label, rel_centroids, out = find_somas(img, res)

brain_url = "s3://open-neurodata/brainlit/brain1"
segments_url = "s3://open-neurodata/brainlit/brain1_segments"
ngl_sess = NeuroglancerSession(mip=1, url=brain_url, url_segments=segments_url)
ImgPath = glob.glob('soma_detection/data/brain1/volumes/*.npy')
hit = np.zeros(1)    # True number = predicted number\n",
miss = np.zeros(1)   # True number > prediceted number\n",
FalsePos = np.zeros(1)   # True number < predicted number\n",
TotalSoma = 0   # initiate total number of soma\n",

NumSub = len(ImgPath)
for item in range(NumSub):
    print(item)
    volume_coords = np.array(os.path.basename(ImgPath[item]).replace('.npy','').split("_")).astype(float).round().astype(int).tolist()
    res = ngl_sess.cv_segments.scales[ngl_sess.mip]["resolution"]
    _min = volume_coords[:3]
    _max = volume_coords[3:]
    vox_min = np.round(np.divide(_min, res)).astype(int)
    vox_max = np.round(np.divide(_max, res)).astype(int)
    # obtain soma coordinate\n",
    contained_coords = np.load(ImgPath[item])
    # true number of soma in subvolume\n",
    TrueNumSo = len(contained_coords)
    bbox = Bbox(vox_min, vox_max)
    img = ngl_sess.pull_bounds_img(bounds=bbox)
    # with napari.gui_qt():\n",
    #     viewer = napari.Viewer(ndisplay=3)\n",
    #     napari.view_image(img)\n",