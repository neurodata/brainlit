from typing import Tuple

import numpy as np
from numpy.core.fromnumeric import ndim
from skimage import filters, morphology, measure
import pandas as pd
from scipy import ndimage

import matplotlib.pyplot as plt


def find_somas(
    volume: np.ndarray, res: np.ndarray
) -> Tuple[int, np.ndarray, np.ndarray]:
    # check resolution is not too low
    if volume.shape[0] < 20 or volume.shape[1] < 20:
        raise ValueError("Resolution of the volume is too low")
    # zoom volume
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
            centroid = np.array([
                int(row["centroid-0"]),
                int(row["centroid-1"]),
                int(row["centroid-2"]),
            ])
            centroid = np.divide(centroid, zoom_factors)
            rel_centroids.append(centroid)
    return num_labels > 0, np.array(rel_centroids), out
