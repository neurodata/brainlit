import pytest
from brainlit.utils import upload_to_neuroglancer
import tifffile as tf
from pathlib import Path
import numpy as np


@pytest.fixture
def volume_info(num_res=1, channel=0):
    top_level = str(Path(__file__).parents[0] / "data_octree")
    (
        ordered_files,
        bin_paths,
        vox_size,
        tiff_dims,
    ) = upload_to_neuroglancer.get_volume_info(top_level, num_res, channel)
    return (
        ordered_files,
        bin_paths,
        vox_size,
        tiff_dims,
    )


# Test volume before uploading chunks
def test_create_image_layer(num_res=1):
    top_level = Path(__file__).parents[0]
    low_res = tf.imread(str(top_level / "data_octree" / "default.0.tif"))
    image_size = low_res.shape[::-1]
    vols = upload_to_neuroglancer.create_image_layer(
        "file://" + str(top_level / "upload"),
        image_size,
        voxel_size=[6173, 6173, 6173],
        num_resolutions=num_res,
    )

    assert len(vols) == num_res
    for i, vol in enumerate(vols):
        assert vol.info["scales"][0]["size"] == [(2 ** i) * j for j in image_size]


def test_get_volume_info(volume_info, num_res=1):
    (ordered_files, bin_paths, vox_size, tiff_dims,) = volume_info
    assert len(ordered_files) == num_res
    assert len(bin_paths) == num_res
    assert len(ordered_files[0]) == 1  # and len(ordered_files[1]) == 8
    # print(vox_size)
    # assert vox_size == [6173, 6173, 6173]  # data specific
    low_res = tf.imread(
        str(Path(__file__).parents[0] / "data_octree" / "default.0.tif")
    )
    image_size = low_res.shape[::-1]
    assert tiff_dims == image_size


# Test stitching ability,
def test_get_data_ranges(volume_info):
    _, bin_paths, _, tiff_dims = volume_info
    for res_bins in bin_paths:
        for bin in res_bins:
            x_curr, y_curr, z_curr = 0, 0, 0
            tree_level = len(bin)
            ranges = upload_to_neuroglancer.get_data_ranges(bin, tiff_dims)
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


# Test cloudvolume attributes (info, shape) for both parallel and non-parallel
def test_upload_chunks_serial(volume_info, num_res=1):
    (ordered_files, bin_paths, vox_size, tiff_dims,) = volume_info
    top_level = Path(__file__).parents[0]
    dir = str(Path(top_level) / "upload" / "test_precomputed_serial")
    vols = upload_to_neuroglancer.create_image_layer(
        "file://" + dir, tiff_dims, vox_size, num_res
    )
    for i in range(num_res):
        upload_to_neuroglancer.upload_chunks(
            vols[i], ordered_files[i], bin_paths[i], parallel=False
        )
    low_res = tf.imread(str(top_level / "data_octree" / "default.0.tif"))
    image_size = low_res.shape[::-1]
    assert (np.squeeze(vols[0][:, :, :]) == low_res.T).all()
    assert np.squeeze(vols[0][:, :, :]).shape == image_size
    # assert np.squeeze(vols[1][:, :, :]).shape == tuple([i * 2 for i in image_size])
    # for file, bin in zip(ordered_files[1], bin_paths[1]):
    #     ranges = upload_to_neuroglancer.get_data_ranges(bin, tiff_dims)
    #     img = np.squeeze(
    #         vols[1][
    #             ranges[0][0] : ranges[0][1],
    #             ranges[1][0] : ranges[1][1],
    #             ranges[2][0] : ranges[2][1],
    #         ]
    #     )
    #     truth = tf.imread(file)
    #     assert (img == truth.T).all()
    #     print("passed")


def test_upload_chunks_parallel(volume_info, num_res=1):
    (ordered_files, bin_paths, vox_size, tiff_dims,) = volume_info
    top_level = Path(__file__).parents[0]
    dir = str(top_level / "upload" / "test_precomputed_parallel")
    par_vols = upload_to_neuroglancer.create_image_layer(
        "file://" + dir, tiff_dims, vox_size, num_res
    )
    for i in range(num_res):
        upload_to_neuroglancer.upload_chunks(
            par_vols[i], ordered_files[i], bin_paths[i], parallel=True
        )

    low_res = tf.imread(str(top_level / "data_octree" / "default.0.tif"))
    image_size = low_res.shape[::-1]
    assert (np.squeeze(par_vols[0][:, :, :]) == low_res.T).all()
    assert np.squeeze(par_vols[0][:, :, :]).shape == image_size
    # assert np.squeeze(par_vols[1][:, :, :]).shape == tuple([i * 2 for i in image_size])
    # for file, bin in zip(ordered_files[1], bin_paths[1]):
    #     ranges = upload_to_neuroglancer.get_data_ranges(bin, tiff_dims)
    #     img = np.squeeze(
    #         par_vols[1][
    #             ranges[0][0] : ranges[0][1],
    #             ranges[1][0] : ranges[1][1],
    #             ranges[2][0] : ranges[2][1],
    #         ]
    #     )
    #     truth = tf.imread(file)
    #     assert (img == truth.T).all()
    #     print("passed")
