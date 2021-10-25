from cloudvolume import CloudVolume
from cloudvolume.exceptions import SkeletonDecodeError
from itertools import islice
from brainlit.utils.session import NeuroglancerSession
from skimage.filters import threshold_otsu, threshold_local
from brainlit.preprocessing import split_frags, rename_states_consecutively, label_points
from brainlit.algorithms.connect_fragments import most_probable_neuron_path
from brainlit.algorithms.connect_fragments import trace_evaluation
from brainlit.viz import Bresenham3D
from cloudvolume import Skeleton

dir = "s3://open-neurodata/brainlit/brain1"
dir_segments = "s3://open-neurodata/brainlit/brain1_segments"

ngl_sess = NeuroglancerSession(mip = 0, url = dir, url_segments=dir_segments)

res = [0.3,0.3,1]
threshold = 0.9

block_size = 101

print(f"downloading")
img, bbox, vox = ngl_sess.pull_vertex_list(2, [0], [1666,1666,500])

print("saving")
io.imsave("/data/tathey1/mouselight/1mm.tif", img)

