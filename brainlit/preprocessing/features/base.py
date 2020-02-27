from abc import abstractmethod
from sklearn.base import BaseEstimator
from brainlit.utils.ngl_pipeline import NeuroglancerSession
import numpy as np
import pandas as pd


class BaseFeatures(BaseEstimator):
    """
    Base class for generating features from precomputed volumes.    
    """

    def __init__(self, url, size=[1, 1, 1], offset=[15, 15, 15]):
        if type(url) is not str:
            raise TypeError("URL must be str")
        self.url = url
        self.size = size
        self.offset = offset

    @abstractmethod
    def _convert_to_features(self, img):
        """
        Computes features from image data.

        Parameters
        ----------
        img : ndarray
            Image data.

        Returns
        -------
        features : ndarray
            Feature data.
        """

    def fit(self, seg_ids, num_verts=None):
        """
        Pulls image and background.

        Parameters
        ----------
        seg_ids : list of ints
            A list of segment indices.
        num_verts : int, optional (default=None)
            If not none, only runs on a set number of vertices.

        Returns
        -------
        df : ndarray
            A dataframe of data.
        """
        df = pd.DataFrame()
        ngl = NeuroglancerSession(self.url)
        ngl_skel = NeuroglancerSession(self.url + "_segments")
        for seg_id in seg_ids:
            segment = ngl_skel.cv.skeleton.get(seg_id)
            if num_verts is not None:
                verts = segment.vertices[:num_verts]
            else:
                verts = segment.vertices
            for v_id, vertex in enumerate(verts):
                img, bounds, voxel = ngl.pull_voxel(
                    seg_id, v_id, self.size[0], self.size[1], self.size[2]
                )
                img_off = ngl.pull_bounds_img(bounds + self.offset)
                features = self._convert_to_features(img)
                features_off = self._convert_to_features(img_off)
                df = df.append(
                    {
                        "Segment": int(seg_id),
                        "Vertex": int(v_id),
                        "Label": 1,
                        "Features": features,
                    },
                    ignore_index=True,
                )
                df = df.append(
                    {
                        "Segment": int(seg_id),
                        "Vertex": int(v_id),
                        "Label": 0,
                        "Features": features_off,
                    },
                    ignore_index=True,
                )
        return df
