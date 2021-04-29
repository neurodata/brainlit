from typing import Tuple

import numpy as np
from numpy.core.fromnumeric import ndim
from skimage import filters, morphology, measure
import pandas as pd
from scipy import ndimage

from brainlit.utils.util import check_type

def find_somas(
    volume: np.ndarray, res: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    r"""Find bright neuron somas in an input volume.

    This simple soma detector assumes that somas are brighter than the
    rest of the objects contained in the input volume.

    To detect somas, these steps are performed:

    #. **Check input volume shape.** This detector requires the `x` and `y` dimensions of the input volumes to be larger than `20` pixels.

    #. **Zoom volume.** We found that this simple soma detector works best when then input volume has size `160 x 160 x 50`. We use `ndimage.zoom <https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html>`_ to scale the input volume size to the desired shape.

    #. **Binarize volume.** We use `Otsu thresholding <https://scikit-image.org/docs/dev/api/skimage.filters.html#skimage.filters.threshold_otsu>`_ to binarize the image.

    #. **Erode the binarized image.** We erode the binarized image with a structuring element which size is directly proportional to the maximum zoom factor applied to the input volume.

    #. **Remove unreasonable connected components.** After erosion, we compute the equivalent diameter `d` of each connected component, and only keep those ones such that `5\mu m \leq d < 21 \mu m`

    #. **Find relative centroids.** Finally, we compute the centroids of the remaining connected components. The centroids are in voxel units, relative to the input volume.

    Parameters
    ----------
    volume : numpy.ndarray
        The 3D image array to run the detector on.

    res : numpy.ndarray
        A `1 x 3` array containing the resolution of each voxel in `nm`.

    Returns
    -------
    label : bool
        A boolean value indicating whether the detector found any somas in the input volume.
    
    rel_centroids : numpy.ndarray
        A `N x 3` list containing the relative voxel positions of the detected somas.
    
    out : numpy.ndarray
        A `160 x 160 x 50` array containing the detection mask.
    """

    res = np.ascontiguousarray(res)
    volume = np.ascontiguousarray(volume)

    check_type(volume, np.ndarray)
    volume_dim = volume.ndim
    if volume_dim != 3:
        raise ValueError("volume must be three-dimensional")
    if volume.shape[0] < 20 or volume.shape[1] < 20:
        raise ValueError("Input volume is too small")

    check_type(res, np.ndarray)

    if np.any(res == 0):
        raise ValueError("Resolution ")

    desired_size = np.array([160, 160, 50])
    zoom_factors = np.divide(desired_size, volume.shape)
    print(res)
    res = np.divide(res, zoom_factors)
    print(res)
    out = ndimage.zoom(volume, zoom=zoom_factors)
    # 1) binarize volume using Otsu's method
    t = filters.threshold_otsu(out)
    out = out > t
    # 2) erode with structuring element proportional to zoom factors
    selem_size = np.amax(np.ceil(zoom_factors)).astype(int)
    clean_selem = morphology.octahedron(selem_size)
    out = morphology.erosion(out, clean_selem)
    # 3) identify connected components
    out, num_labels = morphology.label(out, background=0, return_num=True)
    # 4) remove connected components with diameter not in reasonable range, find centroids of candidate regions
    properties = ["label", "equivalent_diameter", "centroid"]
    props = measure.regionprops_table(out, properties=properties)
    df_props = pd.DataFrame(props)
    rel_centroids = []
    for _, row in df_props.iterrows():
        l = row["label"]
        d = row["equivalent_diameter"]
        dmu = d * np.mean(res[:1]) / 1000
        print(dmu)
        if dmu < 5 or dmu >= 21:
            out[out == l] = 0
            num_labels -= 1
        else:
            centroid = np.array(
                [
                    int(row["centroid-0"]),
                    int(row["centroid-1"]),
                    int(row["centroid-2"]),
                ]
            )
            centroid = np.divide(centroid, zoom_factors)
            rel_centroids.append(centroid)
    return num_labels > 0, np.array(rel_centroids), out
