# Reference: https://github.com/neurodata/mouselight_code/blob/region_growing/src/ngl_pipeline.py

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from cloudvolume import CloudVolume, view
from cloudvolume.lib import Bbox
from cloudvolume.exceptions import InfoUnavailableError
from pathlib import Path
from .swc import read_s3, df_to_graph, get_sub_neuron
import napari
import warnings
import networkx as nx
from typing import Optional, List, Union, Tuple, Literal
from .util import (
    check_type,
    check_size,
    check_precomputed,
    check_iterable_type,
    check_iterable_nonnegative,
)

Bounds = Union[Bbox, Tuple[int, int, int, int, int, int]]


class NeuroglancerSession:
    """Utility class which pulls and pushes data.

    Arguments:
        url: Precompued path either to a file URI or url URI. Defaults to mouselight brain1_2.
        mip: Resolution level to pull and push data at. Defaults to 0, the highest resolution.
        url_segments: Precomputed path to segmentation data. Optional, default None.
        url_annotations: Precomputed path to annotation data. Optonal, default None.

    Attributes:
        url: CloudVolumePrecomputedPath to image data.
        url_segments: CloudVolumePrecomputedPath to segmentation data. Optional, default None.
            Automatically tries precomputed path url+"_segments" if None.
        url_annotations: CloudVolumePrecomputedPath to annotation data. Optional, default None.
            Automatically tries precomputed path url+"_annotation" if None.
        cv (CloudVolumePrecomputed): CloudVolume object for image data.
        cv_segments (CloudVolumePrecomputed): CloudVolume object for segmentation data. Optional, default None.
        cv_annotations (CloudVolumePrecomputed): CloudVolume object for segmentation data. Optional, default None.
        mip: Resolution level.
        chunk_size: The chunk size of the volume at the specified mip, given as (x, y, z).
        scales: The resolution of the volume at the specified mip, given as (x, y, z).
    """

    def __init__(
        self,
        url: str,  #  = "s3://mouse-light-viz/precomputed_volumes/brain1"
        mip: int = 0,
        url_segments: Optional[str] = None,
        url_annotations: Optional[str] = None,
    ):
        check_precomputed(url)
        check_type(mip, int)
        self.url = url
        self.cv = CloudVolume(url, parallel=False)
        if mip < 0 or mip >= len(self.cv.scales):
            raise ValueError(f"{mip} should be between 0 and {len(self.cv.scales)}.")
        self.mip = mip
        self.chunk_size = self.cv.scales[self.mip]["chunk_sizes"][0]
        self.scales = self.cv.scales[self.mip]["resolution"]

        self.url_segments = url_segments
        if url_segments is None:
            try:  # default is to add _segments
                self.cv_segments = CloudVolume(url + "_segments", parallel=False)
                self.url_segments = url + "_segments"
            except InfoUnavailableError:
                warnings.warn(
                    UserWarning(
                        f"Segmentation volume not found at {self.url_segments}, defaulting to None."
                    )
                )
                self.cv_segments = None
        else:
            check_precomputed(url_segments)
            self.cv_segments = CloudVolume(url_segments, parallel=False)

        self.url_annotations = url_annotations
        if url_annotations is None:
            try:  # default is to add _annotation
                self.cv_annotations = CloudVolume(url + "_annotations", parallel=False)
                self.url_annotations = url + "_annotations"
            except InfoUnavailableError:
                warnings.warn(
                    UserWarning(
                        f"Annotation volume not found at {self.url_annotations}, defaulting to None."
                    )
                )
                self.cv_annotations = None
        else:
            check_precomputed(url_annotations)
            self.cv_annotations = CloudVolume(url_annotations, parallel=False)

    def _get_voxel(self, seg_id: int, v_id: int) -> Tuple[int, int, int]:
        """Gets coordinates of segment vertex, in voxel space.

        Arguments:
            seg_id: The id of the segment to use.
            v_id: The id of the vertex to use from the given segment.

        Returns:
            voxel: The voxel coordinates in (x, y, z) voxel space.
        """
        check_type(seg_id, int)
        check_type(v_id, (int, np.integer))
        if self.cv_segments is None:
            raise ValueError("Cannot get voxel without segmentation data")
        seg = self.cv_segments.skeleton.get(seg_id).vertices
        if v_id < 0 or v_id >= len(seg):
            raise ValueError(f"{v_id} should be between 0 and {len(seg)}.")

        vertex = seg[v_id]
        voxel = np.round(
            np.divide(vertex, self.cv_segments.scales[self.mip]["resolution"])
        ).astype(int)
        return voxel

    def set_url_segments(self, seg_url: str):
        """Sets the url_segments and cv_segments attributes.

        Arguments:
            seg_url: CloudvolumePrecomputedPath to segmentation data.
        """
        check_precomputed(seg_url)

        self.url_segments = seg_url
        self.cv_segments = CloudVolume(self.url_segments, parallel=False)

    def set_url_annotations(self, ann_url: str):
        """Sets the url_annotation and cv_annotation attributes.

        Arguments:
            ann_url: CloudvolumePrecomputedPath to annotation data.
        """
        check_precomputed(ann_url)

        self.url_annotations = ann_url
        self.cv_annotations = CloudVolume(self.url_annotations, parallel=False)

    def get_segments(self, seg_id: int, bbox: Optional[Bounds] = None) -> nx.Graph:
        """Get a graph of a segmentation annotation within a bounding box.

        Arguments:
            seg_id  The segement to pull.
            bbox: The bounding box object, default None. If None, uses entire volume.

        Returns:
            G: A subgraph from the specified segment and bounding box.
        """
        check_type(seg_id, int)
        if self.cv_segments is None:
            raise ValueError("Cannot get segments without segmentation data.")

        df = read_s3(self.url_segments, seg_id, self.mip)
        G = df_to_graph(df)
        if bbox is not None:
            if isinstance(bbox, Bbox):
                bbox = bbox.to_list()
            check_iterable_type(bbox, (int, np.integer))
            check_iterable_nonnegative(bbox)
            G = get_sub_neuron(G, [bbox[:3], bbox[3:]])
        return G

    def pull_voxel(
        self, seg_id: int, v_id: int, radius: int = 1
    ) -> Tuple[np.ndarray, Bbox, np.ndarray]:
        """Pull a number of voxels around a specified skeleton vertex.

        Arguments:
            seg_id: ID of the segment to use, depends on data in s3.
            v_id: ID of the vertex to use, depends on the segment.
            radius: Radius of pulled volume around central voxel, in voxels. 
                Optional, default is 1 (3x3 volume is pulled, centered at the vertex).

        Returns:
            img: A 2*nx+1 X 2*ny+1 X 2*nz+1 volume.
            bounds: Bounding box object which contains the bounds of the volume.
            vox_in_img: List of coordinates which locate the initial point in the volume.
        """
        check_type(radius, int)
        if radius < 0:
            raise ValueError(f"{radius} should be nonnegative.")

        voxel = self._get_voxel(seg_id, v_id)  # does type checking for seg_id and v_id
        bounds = Bbox(voxel, voxel)
        seed = bounds.to_list()
        shape = [radius] * 3
        bounds = Bbox(np.subtract(seed[:3], shape), np.add(np.add(seed[3:], shape), 1))
        img = self.pull_bounds_img(bounds)
        # img = self.cv.download(bounds, mip=self.mip)
        vox_in_img = voxel - np.array(bounds.to_list()[:3])
        return np.squeeze(np.array(img)), bounds, vox_in_img

    def pull_vertex_list(
        self,
        seg_id: int,
        v_id_list: List[int],
        buffer: int = 1,
        expand: bool = False,
        source: Literal["image", "annotation"] = "image",
    ) -> Tuple[np.ndarray, Bbox, List[Tuple[int, int, int]]]:
        """Pull a region containing all listed vertices.

        Arguments:
            seg_id: ID of the segment to use, depends on data in s3.
            v_id_list: list of vertex IDs to use.
            buffer: Buffer around the bounding box (in voxels). Default 1, set to 0 if expand is True.
            expand: Flag whether to expand region to closest set of chunks.

        Returns:
            img: The image volume containing all vertices.
            bounds: Bounding box object which contains the bounds of the volume.
            vox_in_img_list: List of coordinates which locate the vertices in the volume.
        """
        check_type(seg_id, int)
        check_iterable_type(v_id_list, (int, np.integer))
        check_type(buffer, int)
        if buffer < 0:
            raise ValueError(f"Buffer {buffer} shouild not be negative.")
        check_type(expand, bool)
        if expand:
            buffer = 0
        buffer = [buffer] * 3
        if source == "annotation" and self.cv_annotations is None:
            raise ValueError("Cannot get annotation data without annotation source.")

        voxel_list = [self._get_voxel(seg_id, i) for i in v_id_list]
        if len(voxel_list) == 1:  # edge case of 1 vertex
            bounds = Bbox(voxel_list[0] - buffer, voxel_list[0] + buffer + 1)
        else:
            voxel_list = np.array(voxel_list)
            lower = list(np.min(voxel_list, axis=0) - buffer)
            higher = list(np.max(voxel_list, axis=0) + buffer + 1)
            bounds = Bbox(lower, higher)

        if expand:
            bounds = bounds.expand_to_chunk_size(self.chunk_size)
        lower = bounds.to_list()[:3]
        if source == "image":
            img = self.pull_bounds_img(bounds)
        else:
            img = self.pull_bounds_seg(bounds)
        vox_in_img_list = np.array(voxel_list) - lower
        return img, bounds, vox_in_img_list

    def pull_chunk(
        self,
        seg_id: int,
        v_id: int,
        radius: int = 0,
        source: Literal["image", "annotation"] = "image",
    ) -> Tuple[np.ndarray, Bbox, Tuple[int, int, int]]:
        """Pull a number of chunks around a specified skeleton vertex.

        Arguments:
            seg_id: ID of the segment to use, depends on data in s3.
            v_id: ID of the vertex to use, depends on the segment.
            radius: Radius of pulled volume around central chunk, in chunks. 
                Optional, default is 0 (single chunk which contains the voxel).

        Returns:
            img: A chunk_size[0]*2*nx X chunk_size[1]*2*ny X chunk_size[2]*2*nz volume.
            bounds: Bounding box object which contains the bounds of the volume.
            vox_in_img: List of coordinates which locate the initial point in the volume.
        """
        check_type(seg_id, int)
        check_type(v_id, int)
        check_type(radius, int)
        if radius < 0:
            raise ValueError(f"Radius of {radius} should be nonnegative.")
        if source == "annotation" and self.cv_annotations is None:
            raise ValueError("Cannot get annotation data without annotation source.")

        voxel = self._get_voxel(seg_id, v_id)
        bounds = Bbox(voxel, voxel).expand_to_chunk_size(self.chunk_size)
        seed = bounds.to_list()
        shape = [
            self.chunk_size[0] * radius,
            self.chunk_size[1] * radius,
            self.chunk_size[2] * radius,
        ]
        bounds = Bbox(np.subtract(seed[:3], shape), np.add(seed[3:], shape))
        if source == "image":
            img = self.pull_bounds_img(bounds)
        else:
            img = self.pull_bounds_seg(bounds)
        vox_in_img = voxel - np.array(bounds.to_list()[:3])
        return np.squeeze(np.array(img)), bounds, vox_in_img

    def pull_bounds_img(self, bounds: Bounds) -> np.ndarray:
        """Pull a volume around a specified bounding box. Works on image channels.

        Arguments:
            bounds: Bounding box, or tuple containing (x0, y0, z0, x1, y1, z1) bounds.

        Returns:
            img: Volume pulled according to the bounding box.
        """
        if isinstance(bounds, Bbox):
            bounds = bounds.to_list()
        check_iterable_type(bounds, (int, np.integer))
        check_iterable_nonnegative(bounds)
        img = self.cv.download(Bbox(bounds[:3], bounds[3:]), mip=self.mip)
        return np.squeeze(np.array(img))

    def pull_bounds_seg(self, bounds: Bounds) -> np.ndarray:
        """Pull a volume around a specified bounding box.
        Works on annotation channels.

        Arguments:
            bounds: Bounding box, or tuple containing (x0, y0, z0, x1, y1, z1) bounds.

        Returns:
            img: Volume pulled according to the bounding box.
        """
        if isinstance(bounds, Bbox):
            bounds = bounds.to_list()
        check_iterable_type(bounds, (int, np.integer))
        check_iterable_nonnegative(bounds)
        if self.cv_annotations is None:
            raise ValueError("Cannot pull from undefined annotation layer.")

        img = self.cv_annotations[Bbox(bounds[:3], bounds[3:])]
        return np.squeeze(np.array(img))

    def push(
        self, img: np.ndarray, bounds: Bounds,
    ):
        """Push a volume to an annotation channel.

        Arguments:
            img : Volume to push
            bounds : Bounding box or tuple containing (x0, y0, z0, x1, y1, z1) bounds.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Image should be numpy array..")
        if (img == 0).all():
            raise ValueError(f"Should not push an empty volume of all 0.")
        if isinstance(bounds, Bbox):
            bounds = bounds.to_list()
        check_iterable_type(bounds, (int, np.integer))
        check_iterable_nonnegative(bounds)
        if self.cv_annotations is None:
            raise ValueError("Cannot pull from undefined annotation layer.")
        self.cv_annotations[Bbox(bounds[:3], bounds[3:])] = img.astype("uint64")
