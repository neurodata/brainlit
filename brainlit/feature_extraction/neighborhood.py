import numpy as np
import brainlit
from brainlit.preprocessing import preprocess, image_process
from scipy import ndimage as ndi
from pathlib import Path
import pandas as pd
from itertools import product
from typing import List, Optional, Dict, Tuple

from .base import BaseFeatures


class NeighborhoodFeatures(BaseFeatures):
    """Computes features based off neighborhood properties.

    Arguments:
        url: Precompued path either to a file URI or url URI of image data.
        size: A size hyperparameter. In Neighborhoods, this is the radius.
        offset: Added to the coordinates of a positive sample to generate a negative sample.
        segment_url: Precompued path either to a file URI or url URI of segmentation data.

    Attributes:
        url: CloudVolumePrecomputedPath to image data.
        size: A size hyperparameter. In Neighborhoods, this is the radius.
        offset: Added to the coordinates of a positive sample to generate a negative sample.
        download_time: Tracks time taken to download the data.
        conversion_time: Tracks time taken to convert data to features.
        write_time: Tracks time taken to write features to files.
        segment_url: CloudVolumePrecomputedPath to segmentation data.
    """

    def __init__(
        self,
        url: str,
        size: List[int] = [1, 1, 1],
        offset: List[int] = [15, 15, 15],
        segment_url: Optional[str] = None,
    ):
        super().__init__(url=url, size=size, offset=offset, segment_url=segment_url)

    def _convert_to_features(self, img: np.ndarray) -> Dict:
        """Computes features from image data by flattening the image.
        """
        return dict(enumerate(img.flatten()))


def subsample(
    arr: np.ndarray, orig_shape: Tuple[int, int, int], dest_shape: Tuple[int, int, int]
) -> np.ndarray:
    """Subsamples a flattened neighborhood to a smaller flattened neighborhood.
    
    Arguments:
        arr: The flattened array
        orig_shape: The original shape of the array before flattening
        dest_shape: The desired shape of the array before flattening
    """
    start = np.subtract(orig_shape, dest_shape) // 2
    end = start + dest_shape
    if len(orig_shape) is 2:
        idx = np.ravel_multi_index(
            (np.mgrid[start[0] : end[0], start[1] : end[1]].reshape(2, -1)), orig_shape
        )
    elif len(orig_shape) is 3:
        idx = np.ravel_multi_index(
            (
                np.mgrid[
                    start[0] : end[0], start[1] : end[1], start[2] : end[2]
                ].reshape(3, -1)
            ),
            orig_shape,
        )
    return arr[idx]
