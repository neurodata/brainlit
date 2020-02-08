# Reference: https://github.com/neurodata/mouselight_code/blob/region_growing/src/ngl_pipeline.py

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from cloudvolume import CloudVolume, view
from cloudvolume.lib import Bbox

class NeuroglancerSession:
    def __init__(self, url="s3://mouse-light-viz/precomputed_volumes/brain1", mip=1):
        self.seed = 1111
        self.url = url
        self.cv = CloudVolume(self.url, parallel=True)
        self.mip = mip
        self.chunk_size = self.cv.info['scales'][self.mip]['chunk_sizes'][0]
        self.scales = self.cv.scales[self.mip]["resolution"]

    def get_voxel(self, seg_id, v_id):
        skeleton_url = "s3://mouse-light-viz/precomputed_volumes/brain1_segments"
        cv_skel = CloudVolume(skeleton_url, mip=self.mip)
        skel = cv_skel.skeleton.get(seg_id)
        vertex = skel.vertices[v_id]
        voxel = np.round(np.divide(vertex, self.scales)).astype(int)
        return voxel

    def pull_voxel(self, seg_id, v_id, nx=0, ny=0, nz=0, expand=True):
        voxel = self.get_voxel(seg_id, v_id)
        bounds = Bbox(voxel, voxel).expand_to_chunk_size(self.chunk_size)
        seed = bounds.to_list()
        shape = [self.chunk_size[0]*nx, self.chunk_size[1]*ny, self.chunk_size[2]*nz]
        bounds = Bbox(np.subtract(seed[:3], shape), np.add(seed[3:], shape))
        img = self.cv.download(bounds, mip=self.mip)
        vox_in_img = voxel - np.array(bounds.to_list()[:3])
        return np.squeeze(np.array(img)), bounds, vox_in_img

    def pull_bounds_img(self, bounds):
        img = self.cv.download(bounds, mip=self.mip)
        return np.squeeze(np.array(img))

    def pull_bounds_seg(self, bounds):
        img = self.cv[bounds]
        return np.squeeze(np.array(img))

    def push(self, img, bounds):
        self.cv[bounds] = img.astype("uint64")
