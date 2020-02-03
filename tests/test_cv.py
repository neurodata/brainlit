import pytest
import numpy as np
from cloudvolume.lib import mkdir, clamp, xyzrange, Vec, Bbox, min2, max2
from cloudvolume.datasource.precomputed.metadata import PrecomputedMetadata

# import brainlight
# from brainlight.utils.ngl_pipeline import NeuroglancerSession
from cloudvolume import CloudVolume


def test_pull():
    # ngl = NeuroglancerSession()
    # meta = PrecomputedMetadata("s3://mouse-light-viz/precomputed_volumes/brain1_seg")
    # print(meta)
    # print(meta.chunk_size(0))
    vol_up = CloudVolume("s3://mouse-light-viz/precomputed_volumes/brain1_seg")
    print(vol_up.info)
    print(vol_up.info.chunk_size[0])
    offset = (0, 0, 0)
    arr = np.zeros((10, 10, 10, 1)).astype("uint64")
    shape = Vec(*arr.shape)[:3]
    offset = Vec(*offset)[:3]
    bounds = Bbox(offset, shape + offset)
    # print(bounds)
    # print(bounds.expand_to_chunk_size())
    vol_up[10:20, 10:20, 10:20] = arr


"""
vol = CloudVolume('s3://mouse-light-viz/precomputed_volumes/brain1', parallel=True, progress=True)
vol_up = CloudVolume('s3://mouse-light-viz/precomputed_volumes/brain1_seg', parallel=False, progress=True)
voltest = CloudVolume('s3://mouse-light-viz/precomputed_volumes/braintest', parallel=False, progress=True)
chunk = vol_up.info['scales'][0]['chunk_sizes'][0]
x, y, z = chunk[0], chunk[1], chunk[2]
a = 0
b = 1
initial = vol_up[a*x:b*x,a*y:b*y,a*z:b*z] # read from brain1_seg
initial2 = vol[a*x:b*x,a*y:b*y,a*z:b*z] # read from brain1_seg
#time.sleep(5)
print("read")
#initial2 = voltest[a*x:b*x,a*y:b*y,a*z:b*z] # read from braintest
#print("read")
#vol_up[a*x:b*x,a*y:b*y,a*z:b*z] = initial2# read from brain1_seg
#print("write")

vol_up[a*x:b*x,a*y:b*y,a*z:b*z] = initial# read from braintest
print('wrote')

#initial = voltest[a*x:b*x,a*y:b*y,a*z:b*z] # read from braintest
voltest[a*x:b*x,a*y:b*y,a*z:b*z] = initial2# read from braintest


#vol_up[10:20,10:20,10:20] = initial # Upload an entire image stack from a numpy array to the cloud
#back_down = vol_up[10:20,10:20,10:20]
#assert(initial==back_down)

#print(initial)
#print(back_down)
#print(image)
"""
