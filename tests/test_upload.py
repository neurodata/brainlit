import pytest
from brainlit.utils import upload
import tifffile as tf
from pathlib import Path
import numpy as np
from glob import glob


@pytest.fixture
def volume_info(num_res=1, channel=0):
    top_level = Path(__file__).parents[1] / "data"
    (ordered_files, bin_paths, vox_size, tiff_dims, origin,) = upload.get_volume_info(
        str(top_level / "data_octree"), num_res, channel
    )
    return (
        ordered_files,
        bin_paths,
        vox_size,
        tiff_dims,
        origin,
        top_level,
    )


def test_get_volume_info(volume_info):
    ordered_files, bin_paths, vox_size, tiff_dims, origin, top_level = volume_info
    assert len(ordered_files[0]) == 1  # and len(ordered_files[1]) == 8
    low_res = tf.imread(str(top_level / "data_octree" / "default.0.tif"))
    image_size = low_res.shape[::-1]
    assert tiff_dims == image_size


def test_create_image_layer(volume_info):
    _, b, vox_size, tiff_dims, _, top_level = volume_info
    vols = upload.create_cloud_volume(
        "file://" + str(top_level / "test_upload"),
        tiff_dims,
        vox_size,
        num_resolutions=len(b),
        layer_type="image",
    )

    assert len(vols) == len(b)
    for i, vol in enumerate(vols):
        assert vol.info["scales"][0]["size"] == [(2 ** i) * j for j in tiff_dims]


def test_create_segmentation_layer(volume_info):
    _, b, vox_size, tiff_dims, _, top_level = volume_info
    vols = upload.create_cloud_volume(
        "file://" + str(top_level / "test_upload_segments"),
        tiff_dims,
        vox_size,
        num_resolutions=len(b),
        layer_type="segmentation",
    )

    assert len(vols) == 1
    for i, vol in enumerate(vols):
        assert vol.info["scales"][0]["size"] == [(2 ** i) * j for j in tiff_dims]


def test_get_data_ranges(volume_info):
    _, bin_paths, _, tiff_dims, _, _ = volume_info
    for res_bins in bin_paths:
        for bin in res_bins:
            x_curr, y_curr, z_curr = 0, 0, 0
            tree_level = len(bin)
            ranges = upload.get_data_ranges(bin, tiff_dims)
            if tree_level == 0:
                assert ranges == ([0, 528], [0, 400], [0, 208])
            else:
                bin = bin[0]
                scale_factor = 1
                x_curr += int(bin[2]) * 528 * scale_factor
                y_curr += int(bin[1]) * 400 * scale_factor
                z_curr += int(bin[0]) * 208 * scale_factor
                assert ranges[0] == [x_curr, x_curr + 528]
                assert ranges[1] == [y_curr, y_curr + 400]
                assert ranges[2] == [z_curr, z_curr + 208]


def test_upload_chunks_serial(num_res=1):
    top_level = Path(__file__).parents[1] / "data"
    input = str(top_level / "data_octree")
    dir = "file://" + str(top_level / "test_upload" / "serial")
    upload.upload_volumes(input, dir, num_mips=num_res)


def test_upload_chunks_parallel(num_res=1):
    top_level = Path(__file__).parents[1] / "data"
    input = str(top_level / "data_octree")
    dir = "file://" + str(Path(top_level) / "test_upload" / "parallel")
    upload.upload_volumes(input, dir, num_mips=num_res, parallel=True)


### segmentation ###


def test_create_skel_segids(volume_info):
    _, _, _, _, origin, top_level = volume_info
    input = str(top_level / "data_octree")
    swc_dir = glob(f"{input}/*consensus-swcs")[0]
    skels, segids = upload.create_skel_segids(swc_dir, origin)
    assert segids[0] == 2
    assert len(skels[0].vertices) > 0


def test_upload_segments(volume_info, num_res=1):
    top_level = Path(__file__).parents[1] / "data"
    input = str(top_level / "data_octree")
    dir = "file://" + str(top_level / "test_upload_segments" / "serial")
    upload.upload_segments(input, dir, num_mips=num_res)
    # unsure how to verify via tests
    assert True
