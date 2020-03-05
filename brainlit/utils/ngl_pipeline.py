# Reference: https://github.com/neurodata/mouselight_code/blob/region_growing/src/ngl_pipeline.py

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from cloudvolume import CloudVolume, view
from cloudvolume.lib import Bbox


class NeuroglancerSession:
    """
    Utility class which pulls and pushes data.

    Parameters
    ----------
    url : string
        URL of the s3 bucket to pull from and push to.

    mip : int, optional (default=1)
        Resolution level to pull and push at. 0 is the highest resolution.

    Attributes
    ----------
    url : string
        URL of the s3 bucket to pull from and push to.

    cv : CloudVolume object
        CloudVolume object instantiated with the specified URL.

    mip : int
        Resolution level.

    chunk_size : list
        The chunk size of the volume at the specified mip, given as [x, y, z].

    scales : list
        The resolution of the volume at the specified mip, given as [x, y, z].
    """

    def __init__(self, url="s3://mouse-light-viz/precomputed_volumes/brain1", mip=1):
        self.url = url
        self.cv = CloudVolume(self.url, parallel=True)
        self.mip = mip
        self.chunk_size = self.cv.info["scales"][self.mip]["chunk_sizes"][0]
        self.scales = self.cv.scales[self.mip]["resolution"]

    def _get_voxel(self, seg_id, v_id):
        skeleton_url = "s3://mouse-light-viz/precomputed_volumes/brain1_segments"
        cv_skel = CloudVolume(skeleton_url, mip=self.mip)
        skel = cv_skel.skeleton.get(seg_id)
        vertex = skel.vertices[v_id]
        voxel = np.round(np.divide(vertex, self.scales)).astype(int)
        return voxel

    def pull_voxel(self, seg_id, v_id, nx=1, ny=1, nz=1):
        """
        Pull a number of voxels around a specified skeleton vertex

        Parameters
        ----------
        seg_id : int
            ID of the segment to use, depends on data in s3.

        v_id : int
            ID of the vertex to use, depends on the segment.

        nx : int, optional (default=1)
            Number of voxels to pull on either side of the seed in x.

        ny : int, optional (default=1)
            Number of voxels to pull on either side of the seed in y.

        nz : int, optional (default=1)
            Number of voxels to pull on either side of the seed in z.

        Returns
        -------
        img : ndarray
            A 2*nx+1 X 2*ny+1 X 2*nz+1 volume.

        bounds : Bbox object
            Bounding box object which contains the bounds of the volume.

        vox_in_img : ndarray
            List of coordinates which locate the initial point in the volume.
        """
        voxel = self._get_voxel(seg_id, v_id)
        bounds = Bbox(voxel, voxel)
        seed = bounds.to_list()
        shape = [nx, ny, nz]
        bounds = Bbox(np.subtract(seed[:3], shape), np.add(np.add(seed[3:], shape), 1))
        img = self.cv.download(bounds, mip=self.mip)
        vox_in_img = voxel - np.array(bounds.to_list()[:3])
        return np.squeeze(np.array(img)), bounds, vox_in_img

    def pull_vertex_list(self, seg_id, v_id_list, buffer=[0, 0, 0], expand=False):
        """
        Pull a number of chunks around a specified skeleton vertex

        Parameters
        ----------
        seg_id : int
            ID of the segment to use, depends on data in s3.

        v_id_list : list of ints
            list of vertex IDs to use.

        buffer : list of ints, optional (default=[0, 0, 0])
            Buffer around the bounding box of seed vertices (on lower and higher bound).

        expand : bool, optional (default=False)
            Flag whether to expand region to closest combination of chunks.

        Returns
        -------
        img : ndarray
            The image volume containing all vertices.

        bounds : Bbox object
            Bounding box object which contains the bounds of the volume.

        vox_in_img_list : ndarray, shape nx3
            List of coordinates which locate the vertices in the volume.
        """
        voxel_list = np.array([self._get_voxel(seg_id, i) for i in v_id_list])
        lower = list(np.min(voxel_list, axis=0) - buffer)
        higher = list(np.max(voxel_list, axis=0) + buffer)
        bounds = Bbox(lower, higher)
        if expand:
            bounds = bounds.expand_to_chunk_size(self.chunk_size)
            lower = bounds.to_list()[:3]
        img = self.pull_bounds_img(bounds)
        vox_in_img_list = voxel_list - lower
        return img, bounds, vox_in_img_list

    def pull_chunk(self, seg_id, v_id, nx=0, ny=0, nz=0):
        """
        Pull a number of chunks around a specified skeleton vertex

        Parameters
        ----------
        seg_id : int
            ID of the segment to use, depends on data in s3.

        v_id : int
            ID of the vertex to use, depends on the segment.

        nx : int, optional (default=0)
            Number of chunks to pull on either side of the main chunk in x.

        ny : int, optional (default=0)
            Number of chunks to pull on either side of the main chunk in y.

        nz : int, optional (default=0)
            Number of chunks to pull on either side of the main chunk in z.

        Returns
        -------
        img : ndarray
            A chunk_size[0]*2*nx X chunk_size[1]*2*ny X chunk_size[2]*2*nz
            volume.

        bounds : Bbox object
            Bounding box object which contains the bounds of the volume.

        vox_in_img : ndarray
            List of coordinates which locate the initial point in the volume.
        """
        voxel = self._get_voxel(seg_id, v_id)
        bounds = Bbox(voxel, voxel).expand_to_chunk_size(self.chunk_size)
        seed = bounds.to_list()
        shape = [
            self.chunk_size[0] * nx,
            self.chunk_size[1] * ny,
            self.chunk_size[2] * nz,
        ]
        bounds = Bbox(np.subtract(seed[:3], shape), np.add(seed[3:], shape))
        img = self.cv.download(bounds, mip=self.mip)
        vox_in_img = voxel - np.array(bounds.to_list()[:3])
        return np.squeeze(np.array(img)), bounds, vox_in_img

    def pull_bounds_img(self, bounds):
        """
        Pull a volume around a specified bounding box. Works on img channels.

        Parameters
        ----------
        bounds : Bbox object
            Tuple containing (x0, y0, z0, x1, y1, z1) bounds

        Returns
        -------
        img : ndarray
            pulled volume
        """
        img = self.cv.download(bounds, mip=self.mip)
        return np.squeeze(np.array(img))

    def pull_bounds_seg(self, bounds):
        """
        Pull a volume around a specified bounding box.
        Works on annotation channels.

        Parameters
        ----------
        bounds : Bbox object
            Tuple containing (x0, y0, z0, x1, y1, z1) bounds

        Returns
        -------
        img : ndarray
            pulled volume
        """
        img = self.cv[bounds]
        return np.squeeze(np.array(img))

    def push(self, img, bounds):
        """
        Push a volume.

        Parameters
        ----------
        img : ndarray
            Volume to push

        bounds : Bbox object
            Tuple containing (x0, y0, z0, x1, y1, z1) bounds
        """
        self.cv[bounds] = img.astype("uint64")
