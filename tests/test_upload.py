import numpy as np
from cloudvolume import CloudVolume, Skeleton, storage
from pathlib import Path
import tifffile as tf
from joblib import Parallel, delayed, cpu_count
from brainlit.utils import upload_to_neuroglancer
import os

dir = os.path.dirname(os.path.abspath(__file__))
top_level = os.path.join(dir,'data/')

num_res = 2
test_vox_size = [10.5,9.78,8.99]
volume_size=[528,400,208]
#vol = CloudVolume(
#    "s3://open-neurodata/kasthuri/kasthuri11/image", mip=0, use_https=True
#)
#print(vol.info)

# Test volume before uploading chunks
def test_create_image_layer():
    vols = upload_to_neuroglancer.create_image_layer('file://'+ dir + '/test_precomputed/', test_vox_size, num_res)
    print(vols[0].info['scales'][0])
    print(vols[1].info['scales'][1])
    assert len(vols) == num_res
    for r in range(num_res):
        assert vols[r].info['scales'][r]['size'] == [dim*2**(num_res-r-1) for dim in volume_size]
        assert vols[r].mip == num_res-r-1
        #assert vols[r].info['scales'][r]['resolution'] == [res*2**r for res in test_vox_size]

# Make some empty directories?
def test_get_file_paths():
    ordered_files, bin_paths = upload_to_neuroglancer.get_file_paths(top_level, num_res, 0)
    print(ordered_files, bin_paths)
    assert len(ordered_files) == num_res
    assert len(bin_paths) == num_res

# Test cloudvolume attributes (info, shape) for both parallel and non-parallel
def test_upload_chunks():
    assert True


# Test stitching ability,
def test_get_data_ranges():
    bin_paths = upload_to_neuroglancer.get_file_paths(top_level, num_res, 0)[1]
    print(bin_paths[1])
    for res_bins in bin_paths:
        print(res_bins)
        ranges = upload_to_neuroglancer.get_data_ranges(res_bins, volume_size)
    assert True
#test_create_image_layer()
test_get_file_paths()
#test_get_data_ranges()
