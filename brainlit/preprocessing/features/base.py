from abc import abstractmethod
from sklearn.base import BaseEstimator
from brainlit.utils.session import NeuroglancerSession
import numpy as np
import pandas as pd
import time
from cloudvolume import CloudVolume
import feather
from joblib import Parallel, delayed


class BaseFeatures(BaseEstimator):
    """
    Base class for generating features from precomputed volumes.    
    """

    def __init__(self, url, size=[1, 1, 1], offset=[15, 15, 15], segment_url=None):
        if type(url) is not str:
            raise TypeError("URL must be str")
        self.url = url
        self.size = size
        self.offset = offset
        self.segment_url = segment_url

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

    def fit(
        self,
        seg_ids,
        num_verts=None,
        file_path=None,
        batch_size=10000,
        start_seg=None,
        start_vert=0,
        include_neighborhood=False,
        n_jobs=1,
    ):
        """
        Pulls image and background.

        Parameters
        ----------
        seg_ids : list of ints
            A list of segment indices.

        num_verts : int, optional (default=None)
            If not none, only runs on a set number of vertices.
        
        file_path : str
            If not none, then the extracted data will be written directly
            into a feather binary file. The file_path specifies the prefix
            of the file.

        batch_size : int
            Size of each batch of data to be loaded/written. Is only
            used when file_path is not none.
        
        start_seg : int
            Specifies which segment in the seg_ids list to start at.
        
        start_vert : int
            Specifies which vertex of the first seg_id to start at
        
        include_neighborhood : boolean
            If extracting linear features, specifies if the general
            neighborhood should be extracted as well.
        
        n_jobs : int
            Number of cores to use. -1 to use all available cores.

        Returns
        -------
        df : ndarray
            A dataframe of data.
        """
        voxel_dict = {}
        counter = 0
        batch_id = 0
        ngl = NeuroglancerSession(self.url, segment_url=self.segment_url)
        # if self.segment_url is None:
        #     ngl_skel = NeuroglancerSession(self.url)
        # else:
        #     ngl_skel = NeuroglancerSession(self.url, self.segment_url)

        if start_seg is not None:
            seg_ids = seg_ids[seg_ids.index(start_seg) :]

        if file_path is None:
            return self._serial_processing(
                seg_ids, ngl, num_verts, start_vert, include_neighborhood
            )
        else:
            if n_jobs == 1:
                self._serial_processing(
                    seg_ids,
                    ngl,
                    num_verts,
                    start_vert,
                    include_neighborhood,
                    True,
                    batch_size,
                    file_path,
                )
            else:
                start_vertices = [0] * len(seg_ids)
                start_vertices[0] = start_vert
                par = Parallel(n_jobs=n_jobs)
                par(
                    delayed(self._parallel_processing)(
                        seg_id,
                        ngl,
                        ngl_skel,
                        num_verts,
                        start_vertices[i],
                        include_neighborhood,
                        batch_size,
                        file_path,
                    )
                    for i, seg_id in enumerate(seg_ids)
                )

    def _serial_processing(
        self,
        seg_ids,
        ngl,
        num_verts,
        start_vert,
        include_neighborhood,
        write=False,
        batch_size=None,
        file_path=None,
    ):
        voxel_dict = {}
        counter = 0
        batch_id = 0

        for seg_id in seg_ids:
            if self.segment_url is None:
                segment = ngl.cv.skeleton.get(seg_id)
            else:
                cv_skel = CloudVolume(self.segment_url)
                segment = cv_skel.skeleton.get(seg_id)
            if num_verts is not None:
                verts = segment.vertices[start_vert:num_verts]
            else:
                verts = segment.vertices[start_vert:]
            start_vert = 0
            for v_id, vertex in enumerate(verts):
                img, bounds, voxel = ngl.pull_voxel(
                    seg_id, v_id, self.size[0], self.size[1], self.size[2]
                )
                img_off = ngl.pull_bounds_img(bounds + self.offset)
                features = self._convert_to_features(img, include_neighborhood)
                features_off = self._convert_to_features(img_off, include_neighborhood)
                voxel_dict[counter] = {
                    **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 1},
                    **features,
                }
                counter += 1
                voxel_dict[counter] = {
                    **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 0},
                    **features_off,
                }
                counter += 1
                if write:
                    if counter % batch_size == 0 or (counter + 1) % batch_size == 0:
                        df = pd.DataFrame.from_dict(voxel_dict, "index")
                        path = (
                            file_path
                            + str(batch_id * batch_size)
                            + "_"
                            + str((batch_id + 1) * batch_size)
                            + "_"
                            + str(seg_id)
                            + "_"
                            + str(v_id)
                            + ".feather"
                        )
                        feather.write_dataframe(df, path)
                        voxel_dict = {}
                        batch_id += 1
        if write:
            if not (counter % batch_size == 0 or (counter + 1) % batch_size == 0):
                df = pd.DataFrame.from_dict(voxel_dict, "index")
                path = (
                    file_path
                    + str(batch_id * batch_size)
                    + "_"
                    + str(counter)
                    + "_"
                    + str(seg_id)
                    + "_"
                    + str(v_id)
                    + ".feather"
                )
                feather.write_dataframe(df, path)
        else:
            df = pd.DataFrame.from_dict(voxel_dict, "index")
            return df

    def _parallel_processing(
        self,
        seg_id,
        ngl,
        num_verts,
        start_vert,
        include_neighborhood,
        batch_size=None,
        file_path=None,
    ):
        voxel_dict = {}
        counter = 0
        batch_id = 0
        if self.segment_url is None:
            segment = ngl.cv.skeleton.get(seg_id)
        else:
            cv_skel = CloudVolume(self.segment_url)
            segment = cv_skel.skeleton.get(seg_id)
        if num_verts is not None:
            verts = segment.vertices[start_vert:num_verts]
        else:
            verts = segment.vertices[start_vert:]
        for v_id, vertex in enumerate(verts):
            img, bounds, voxel = ngl.pull_voxel(
                seg_id, v_id, self.size[0], self.size[1], self.size[2]
            )
            img_off = ngl.pull_bounds_img(bounds + self.offset)
            features = self._convert_to_features(img, include_neighborhood)
            features_off = self._convert_to_features(img_off, include_neighborhood)
            voxel_dict[counter] = {
                **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 1},
                **features,
            }
            counter += 1
            voxel_dict[counter] = {
                **{"Segment": int(seg_id), "Vertex": int(v_id), "Label": 0},
                **features_off,
            }
            counter += 1
            if counter % batch_size == 0 or (counter + 1) % batch_size == 0:
                df = pd.DataFrame.from_dict(voxel_dict, "index")
                path = (
                    file_path
                    + str(batch_id * batch_size)
                    + "_"
                    + str((batch_id + 1) * batch_size)
                    + "_"
                    + str(seg_id)
                    + "_"
                    + str(v_id)
                    + ".feather"
                )
                feather.write_dataframe(df, path)
                voxel_dict = {}
                batch_id += 1
        if not (counter % batch_size == 0 or (counter + 1) % batch_size == 0):
            df = pd.DataFrame.from_dict(voxel_dict, "index")
            path = (
                file_path
                + str(batch_id * batch_size)
                + "_"
                + str(counter)
                + "_"
                + str(seg_id)
                + "_"
                + str(v_id)
                + ".feather"
            )
            feather.write_dataframe(df, path)
