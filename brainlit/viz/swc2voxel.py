from typing import Tuple
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import draw
from brainlit.utils.util import check_type


def skeletonize(img: np.ndarray, points: pd.DataFrame) -> np.ndarray:
    """Draw lines between points that are connected to produce binary mask

    Arguments:
        img {3d array} -- image
        points {pandas dataframe} -- dataframe with swc points as output by combine_swc_img.points2voxel

    Returns:
        [3d array] -- binary mask showing skeletonization between points
    """
    mask = 0 * img
    print(points)
    for idx, child in points.iterrows():
        parent_idx = child["parent"]
        if parent_idx != -1 and parent_idx in points["sample"].values:
            parent_pt = points[points["sample"] == [parent_idx]]
            xs, ys, zs = Bresenham3D(
                parent_pt.xvox.item(),
                parent_pt.yvox.item(),
                parent_pt.zvox.item(),
                child.xvox,
                child.yvox,
                child.zvox,
            )

            mask[xs, ys, zs] = 1
    return mask


def Bresenham3D(
    x1: int, y1: int, z1: int, x2: int, y2: int, z2: int
) -> Tuple[list, list, list]:
    """Takes two coordinates and gives the set of coordinates that connects them with a straight line

    Adapted from https://www.geeksforgeeks.org/bresenhams-algorithm-for-3-d-line-drawing/

    Arguments:
        x1 {int} -- first x coodinate
        y1 {int} -- first y coodinate
        z1 {int} -- first z coodinate
        x2 {int} -- second x coodinate
        y2 {int} -- second y coodinate
        z2 {int} -- second z coodinate

    Returns:
        [list] -- list of x coordinate connecting the points
        [list] -- list of y coordinate connecting the points
        [list] -- list of z coordinate connecting the points
    """
    check_type(x1, int)
    check_type(y1, int)
    check_type(z1, int)
    check_type(x2, int)
    check_type(y2, int)
    check_type(z2, int)

    xlist = []
    xlist.append(x1)
    ylist = []
    ylist.append(y1)
    zlist = []
    zlist.append(z1)

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)
    if x2 > x1:
        xs = 1
    else:
        xs = -1
    if y2 > y1:
        ys = 1
    else:
        ys = -1
    if z2 > z1:
        zs = 1
    else:
        zs = -1

    # Driving axis is X-axis"
    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

            xlist.append(x1)
            ylist.append(y1)
            zlist.append(z1)

    # Driving axis is Y-axis"
    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz
            xlist.append(x1)
            ylist.append(y1)
            zlist.append(z1)

    # Driving axis is Z-axis"
    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx
            xlist.append(x1)
            ylist.append(y1)
            zlist.append(z1)
    return xlist, ylist, zlist
