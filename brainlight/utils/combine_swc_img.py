import numpy as np
from pathlib import Path
import pandas as pd
from . import read_octree as octree
from . import read_swc


def overlay_swc(parent_dir, swc_path, channel=0):
    """Most abstracted function to overlay swc on image data
    
    Arguments:
        parent_dir {str} -- path to octree parent
        swc_path {str} -- path to swc
    
    Keyword Arguments:
        channel {int} -- image channel to be displayed (default: {0})
    
    Returns:
        img -- image section
        voxels -- voxel coordinates (within the returned image) of swc points
    """
    tree = octree.octree(parent_dir)

    points, _, _, _ = read_swc.read_swc_offset(swc_path)

    img, start = points2img(tree, points)

    points = points2voxel(tree, points, start)

    return img, points


def points2img(tree, points, channel=0, pad=[2, 2, 2]):
    """Find the image data that surrounds a swc.
    Subsequently, points2voxel() can convert the points to voxel coordinates.
    
    Arguments:
        tree {octree object} -- octree
        points {pandas dataframe} -- dataframe of swc points as output by read_swc.read_swc_offset
    
    Returns:
        img -- image data that surrounds swc
        start -- voxel coordinates of upper left corner
    """
    start, end = read_swc.bbox_vox(points)
    start = tree.space_to_voxel(start) - pad
    end = tree.space_to_voxel(end) + pad

    img = tree.get_interrectangle_voxel(start, end, channel)
    return img, start


def points2voxel(tree, points, start):
    """Find relative voxel coordinates in the image that surrounds the swc. 
    Intended use is right after points2img()
    
    Arguments:
        tree {octree object} -- octree
        points {pandas dataframe} -- dataframe of swc points as output by read_swc.read_swc_offset
        start {3 array} -- coordinate of top left of image
    
    Returns:
        voxels -- numpy array of swc point coordinates
    """
    voxels = np.zeros([points.shape[0], 3], dtype=int)

    for idx, row in points.iterrows():
        voxel = tree.space_to_voxel([row.x, row.y, row.z]) - start
        voxels[idx,] = voxel
    points_new = points.copy()
    points_new["xvox"] = voxels[:, 0]
    points_new["yvox"] = voxels[:, 1]
    points_new["zvox"] = voxels[:, 2]
    return points_new
