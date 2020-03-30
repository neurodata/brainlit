import numpy as np
from skimage import draw
import itertools


def pairwise(iterable):
    # Adapted from https://stackoverflow.com/a/5434936
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def draw_sphere(shape, center, radius):
    """
    Generate a sphere of a radius at a point.
    
    Adapted from https://stackoverflow.com/a/56060957
    
    Parameters
    -------
    shape : tuple
        The shape of output array.
    
    center : tuple
        The coordinates for the center of the sphere.
    
    radius : float
        The radius of the sphere.
    
    Returns
    -------
    sphere : numpy.ndarray
        An binary-valued array including a sphere.
    
    """
    coords = np.ogrid[: shape[0], : shape[1], : shape[2]]
    distance = np.sqrt(
        (coords[0] - center[0]) ** 2
        + (coords[1] - center[1]) ** 2
        + (coords[2] - center[2]) ** 2
    )
    sphere = 1 * (distance <= radius)
    return sphere


def draw_tube(img, vertex0, vertex1, radius):
    """
    Generate a segmentation mask of a tube (series of spheres) connecting known vertices.
    
    Parameters
    -------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.
    
    vertex0 : tuple
        A vertex containing a coordinate within a known segment.
    
    vertex1 : tuple
        A vertex containing a coordinate within a known segment.
    
    radius : float
        The radius of the cylinder.
    
    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.
    
    """
    line = draw.line_nd(vertex0, vertex1, endpoint=True)
    line = np.array(line).T
    seg = np.zeros(img.shape)
    for pt in line:
        s = draw_sphere(img.shape, pt, radius)
        # print(s.sum())
        seg += s

    labels = np.where(seg >= 1, 1, 0)
    return labels


def tubes_seg(img, vertices, radius):
    """
    Generate a segmentation mask of cylinders connecting known vertices.
    
    Parameters
    -------
    img : cloudvolume.volumecutout.VolumeCutout
        The volume to segment.
    
    vertices : list
        The vertices (tuples) each containing a coordinate within a known segment.
    
    radius : float
        The radius of the cylinder.
    
    Returns
    -------
    labels : numpy.ndarray
        An array consisting of the pixelwise segmentation.
    
    """
    output = np.zeros(img.shape)
    for a, b in pairwise(vertices):
        output += draw_tube(img, a, b, radius)
    labels = np.where(output >= 1, 1, 0)
    return labels
