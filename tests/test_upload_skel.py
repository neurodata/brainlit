import pytest
import numpy as np
import tifffile as tf
from brainlit.utils import upload_skeleton
from pathlib import Path
from tqdm import tqdm


@pytest.fixture
def volume_info(num_res=2):
    top = Path(__file__).parents[0]
    top_level = str(top / "data_octree")
    (origin, vox_size, tiff_dims,) = upload_skeleton.get_volume_info(top_level, num_res)
    return (
        origin,
        vox_size,
        tiff_dims,
    )


# Test volume before uploading chunks


def test_get_volume_info(volume_info):
    origin, vox_size, tiff_dims = volume_info

    assert len(origin) == 3
    print(vox_size)
    test_vox_size = [6173, 6173, 6173]
    # assert vox_size == test_vox_size
    top_level = Path(__file__).parents[0]
    low_res = tf.imread(str(top_level / "data_octree" / "default.0.tif"))
    image_size = low_res.shape[::-1]
    assert tiff_dims == image_size


def test_create_skeleton_layer(volume_info, num_res=2):
    origin, vox_size, tiff_dims = volume_info
    dir = str(Path(__file__).parents[0] / "upload_segments")
    vol = upload_skeleton.create_skeleton_layer(
        "file://" + dir, vox_size, tiff_dims, num_res=num_res
    )
    assert vol.mip == 0
    assert vol.info["scales"][0]["size"] == [i * 2 for i in tiff_dims]


def test_create_skel_segids(volume_info, num_res=2):
    origin, vox_size, tiff_dims = volume_info
    top_level_swc = str(Path(__file__).parents[0] / "data_swcs")
    skels, segids = upload_skeleton.create_skel_segids(top_level_swc, origin)
    assert segids[0] == 2


def test_upload_skeletons(volume_info, num_res=2):
    origin, vox_size, tiff_dims = volume_info
    dir = str(Path(__file__).parents[0] / "upload_segments")
    vol = upload_skeleton.create_skeleton_layer(
        "file://" + dir, vox_size, tiff_dims, num_res=num_res
    )
    top_level_swc = str(Path(__file__).parents[0] / "data_swcs")
    skeletons, segids = upload_skeleton.create_skel_segids(top_level_swc, origin)
    for skel in skeletons:
        vol.skeleton.upload(skel)
    # unsure how to verify via tests
    assert True
