import numpy as np
import tifffile as tf
from brainlit.utils import upload_to_neuroglancer
import os

dir = os.path.dirname(os.path.abspath(__file__))
top_level = os.path.join(dir, "data/")
low_res = tf.imread(top_level + "default.0.tif")

num_res = 2
test_vox_size = [10.5, 9.78, 8.99]
volume_size = [528, 400, 208]

# Test volume before uploading chunks
def test_create_image_layer():
    vols = upload_to_neuroglancer.create_image_layer("file://", test_vox_size, num_res)

    assert len(vols) == num_res
    assert vols[0].mip == 1
    assert vols[1].mip == 0
    assert vols[0].info["scales"][0]["size"] == [2 * i for i in volume_size]
    assert vols[1].info["scales"][1]["size"] == volume_size
    # assert vols[r].info['scales'][r]['resolution'] == [res*2**r for res in test_vox_size]


def test_get_file_paths():
    ordered_files, bin_paths = upload_to_neuroglancer.get_file_paths(
        top_level, num_res, 0
    )
    assert len(ordered_files) == num_res
    assert len(bin_paths) == num_res
    assert len(ordered_files[0]) == 1 and len(ordered_files[1]) == 8


# Test stitching ability,
def test_get_data_ranges():
    bin_paths = upload_to_neuroglancer.get_file_paths(top_level, num_res, 0)[1]
    for res_bins in bin_paths:
        for bin in res_bins:
            x_curr, y_curr, z_curr = 0, 0, 0
            tree_level = len(bin)
            ranges = upload_to_neuroglancer.get_data_ranges(bin, volume_size)
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
def test_upload_chunks():
    ordered_files, bin_paths = upload_to_neuroglancer.get_file_paths(
        top_level, num_res, 0
    )

    par_vols = upload_to_neuroglancer.create_image_layer(
        "file://" + dir + "/test_precomputed1/", test_vox_size, num_res
    )
    vols = upload_to_neuroglancer.create_image_layer(
        "file://" + dir + "/test_precomputed2/", test_vox_size, num_res
    )
    upload_to_neuroglancer.upload_chunks(
        par_vols[0], ordered_files[0], bin_paths[0], parallel=True
    )

    for i in range(num_res):
        upload_to_neuroglancer.upload_chunks(
            vols[i], ordered_files[i], bin_paths[i], parallel=False
        )
    assert (np.squeeze(par_vols[0][:, :, :]) == low_res.T).all()
    assert np.squeeze(par_vols[0][:, :, :]).shape == (528, 400, 208)
    # assert np.squeeze(par_vols[1][:, :, :]).shape == (528 * 2, 400 * 2, 208 * 2)

    assert (np.squeeze(vols[0][:, :, :]) == low_res.T).all()
    assert np.squeeze(vols[0][:, :, :]).shape == (528, 400, 208)
    assert np.squeeze(vols[1][:, :, :]).shape == (528 * 2, 400 * 2, 208 * 2)

    for bin in bin_paths[1]:
        ranges = upload_to_neuroglancer.get_data_ranges(bin, volume_size)
        assert (
            np.squeeze(
                vols[1][
                    ranges[0][0] : ranges[0][1],
                    ranges[1][0] : ranges[1][1],
                    ranges[2][0] : ranges[2][1],
                ]
            )
            == low_res.T
        ).all()
