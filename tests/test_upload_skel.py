import numpy as np
import tifffile as tf
from brainlit.utils import upload_skeleton
import os

dir = os.path.dirname(os.path.abspath(__file__))
top_level = os.path.join(dir, "data_octree/")
top_level_swc = os.path.join(dir, "data_swcs/")

num_res = 2
test_vox_size = [6173, 6173, 6173]
image_size = (528, 400, 208)


# Test volume before uploading chunks


def test_get_volume_info():
    origin, vox_size, tiff_dims = upload_skeleton.get_volume_info(top_level, num_res)
    assert vox_size == test_vox_size
    assert tiff_dims == image_size
    assert len(origin) == 3


def test_create_skeleton_layer():
    vol = upload_skeleton.create_skeleton_layer(
        "file://", test_vox_size, image_size, num_res
    )
    assert vol.mip == 0
    assert vol.info["scales"][0]["size"] == [i * 2 for i in image_size]


def test_create_skel_segids():
    origin = upload_skeleton.get_volume_info(top_level, num_res)[0]
    skels, segids = upload_skeleton.create_skel_segids(
        top_level_swc, origin
    )
    assert len(segids) == 2
    assert len(skels) == 2
    for skel in skels:
        assert len(skel.vertices) == 7
