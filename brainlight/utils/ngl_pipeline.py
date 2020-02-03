# Reference: https://github.com/neurodata/mouselight_code/blob/region_growing/src/ngl_pipeline.py

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from cloudvolume import CloudVolume, view
from sklearn.mixture import GaussianMixture
import napari


class NeuroglancerSession:
    def __init__(self, url="s3://mouse-light-viz/precomputed_volumes/brain1", mip=1):
        self.seed = 1111
        self.url = url
        self.mip = mip
        cv = CloudVolume(self.url, progress=True, mip=0, cache=False, parallel=True)
        self.cv = cv
        cv_skel = CloudVolume(
            self.url + "_segments", mip=0, cache=False, parallel=False
        )
        self.cv_skel = cv_skel
        self.img = None
        self.labels = None
        self.chunk_size = (50, 50, 50)
        self.sx = None
        self.sy = None
        self.sz = None

    def _scale(self, verts):
        scales = self.cv.scales[self.mip]["resolution"]
        scaled = verts / scales
        int_scaled = [int(i) for i in scaled]
        return np.array(int_scaled).reshape(-1, 1)

    def get_img(self, seg_id, v_id, sx=5, sy=5, sz=5):
        """Gets image to the right (in each dimension) of the (v_id)-th vertex 
        in the neuron skeleton.
        Returns the image and skeleton
        """
        SEGID = seg_id
        skel = self.cv_skel.skeleton.get(SEGID)
        img = self.cv.download_point(
            self._scale(skel.vertices[v_id]),
            mip=self.mip,
            size=(
                self.chunk_size[0] * sx,
                self.chunk_size[1] * sy,
                self.chunk_size[2] * sz,
            ),
            coord_resolution=self.cv.scales[self.mip]["resolution"],
        )
        self.img = img
        self.sx = sx
        self.sy = sy
        self.sz = sz

        ## TODO: convert skel.vertices into voxel coordinates relative to img
        return img, skel

    def view_img(self, img=None, title="View"):
        if img is None:
            img = self.img
        if img is None:
            raise ValueError(f"Need an image to view it, not {type(img)}")
        else:
            with napari.gui_qt():
                viewer = napari.view_image(np.squeeze(np.array(self.img)))
                if self.labels is not None:
                    viewer.add_labels(self.labels, name="segmentation")
        return

    def _img_to_labels(self, img=None, low=None, up=255):
        if img is None:
            img = self.img
        img_T1 = sitk.GetImageFromArray(np.squeeze(img), isVector=False)
        img_T1_255 = sitk.Cast(sitk.RescaleIntensity(img_T1), sitk.sitkUInt8)
        seed = (
            int(self.chunk_size[2] * self.sz / 2),
            int(self.chunk_size[1] * self.sy / 2),
            int(self.chunk_size[0] * self.sx / 2),
        )
        seg = sitk.Image(img_T1.GetSize(), sitk.sitkUInt8)
        seg.CopyInformation(img_T1)
        seg[seed] = 1
        seg = sitk.BinaryDilate(seg, 1)
        if low is None:
            v = sitk.GetArrayFromImage(img_T1_255)
            flat = v.flatten().reshape(-1, 1)  # img
            gmm = GaussianMixture(n_components=2, random_state=self.seed)
            gmm.fit(flat)
            low = np.divide(
                gmm.means_[0][0] * gmm.covariances_[0][0]
                + gmm.means_[1][0] * gmm.covariances_[1][0],
                gmm.covariances_[0][0] + gmm.covariances_[1][0],
            )
            print("result:", low)
        seg_con = sitk.ConnectedThreshold(
            img_T1_255, seedList=[seed], lower=int(np.round(low)), upper=up
        )
        vectorRadius = (1, 1, 1)
        kernel = sitk.sitkBall
        seg_clean = sitk.BinaryMorphologicalClosing(seg_con, vectorRadius, kernel)
        labels = sitk.GetArrayFromImage(seg_clean)
        return labels

    def add_labels(self, labels=None, low=None):
        if labels is None:
            labels = self._img_to_labels(low=low)
        print("ok")
        if not np.any(labels):
            print("Matrix is all 0!")
        self.labels = labels
        return self.labels

    def push_img(self, img=None, url=None):
        if url is None:
            url = self.url + "_seg"
        vol = CloudVolume(
            url,
            mip=self.mip,
            non_aligned_writes=True,
            fill_missing=True,
            delete_black_uploads=False,
        )
        if img is None:
            img = self.labels
        if img is None:
            raise Exception("you are pushing None")
        bounds = self.img.bounds.to_list()
        vol[
            bounds[0] : bounds[3], bounds[1] : bounds[4], bounds[2] : bounds[5]
        ] = img.astype("uint64")
        return
