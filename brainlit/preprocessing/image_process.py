from re import VERBOSE
import numpy as np
from skimage.measure import label, regionprops
import scipy.ndimage as ndi
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from itertools import product
from typing import List, Optional, Union, Tuple
from brainlit.utils.util import (
    check_type,
    check_iterable_type,
    check_iterable_or_non_iterable_type,
    numerical,
)
import collections
import numbers
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp


def gabor_filter(
    input: np.ndarray,
    sigma: Union[float, List[float]],
    phi: Union[float, List[float]],
    frequency: float,
    offset: float = 0.0,
    output: Optional[Union[np.ndarray, np.dtype, None]] = None,
    mode: str = "reflect",
    cval: float = 0.0,
    truncate: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multidimensional Gabor filter. A gabor filter
    is an elementwise product between a Gaussian
    and a complex exponential.

    Parameters
    ----------
    input : array_like
        The input array.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    phi : scalar or sequence of scalars
        Angles specifying orientation of the periodic complex
        exponential. If the input is n-dimensional, then phi
        is a sequence of length n-1. Convention follows
        https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates.
    frequency : scalar
        Frequency of the complex exponential. Units are revolutions/voxels.
    offset : scalar
        Phase shift of the complex exponential. Units are radians.
    output : array or dtype, optional
        The array in which to place the output, or the dtype of the returned array.
        By default an array of the same dtype as input will be created. Only the real component will be saved
        if output is an array.
    mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        The mode parameter determines how the input array is extended beyond its boundaries.
        Default is ‘reflect’.
    cval : scalar, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    real, imaginary : arrays
        Returns real and imaginary responses, arrays of same
        shape as `input`.

    Notes
    -----
    The multidimensional filter is implemented by creating
    a gabor filter array, then using the convolve method.
    Also, sigma specifies the standard deviations of the
    Gaussian along the coordinate axes, and the Gaussian
    is not rotated. This is unlike
    skimage.filters.gabor, whose Gaussian is
    rotated with the complex exponential.
    The reasoning behind this design choice is that
    sigma can be more easily designed to deal with
    anisotropic voxels.

    Examples
    --------
    >>> from brainlit.preprocessing import gabor_filter
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gabor_filter(a, sigma=1, phi=[0.0], frequency=0.1)
    (array([[ 3,  5,  6,  8,  9],
            [ 9, 10, 12, 13, 14],
            [16, 18, 19, 21, 22],
            [24, 25, 27, 28, 30],
            [29, 30, 32, 34, 35]]),
     array([[ 0,  0, -1,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0, -1,  0,  0],
            [ 0,  0, -1,  0,  0]]))

    >>> from scipy import misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure()
    >>> plt.gray()  # show the filtered result in grayscale
    >>> ax1 = fig.add_subplot(121)  # left side
    >>> ax2 = fig.add_subplot(122)  # right side
    >>> ascent = misc.ascent()
    >>> result = gabor_filter(ascent, sigma=5, phi=[0.0], frequency=0.1)
    >>> ax1.imshow(ascent)
    >>> ax2.imshow(result[0])
    >>> plt.show()
    """
    check_type(input, (list, np.ndarray))
    check_iterable_or_non_iterable_type(sigma, numerical)
    check_iterable_or_non_iterable_type(phi, numerical)
    check_type(frequency, numerical)
    check_type(offset, numerical)
    check_type(cval, numerical)
    check_type(truncate, numerical)

    input = np.asarray(input)

    # Checks that dimensions of inputs are correct
    sigmas = ndi._ni_support._normalize_sequence(sigma, input.ndim)
    phi = ndi._ni_support._normalize_sequence(phi, input.ndim - 1)

    limits = [np.ceil(truncate * sigma).astype(int) for sigma in sigmas]
    ranges = [range(-limit, limit + 1) for limit in limits]
    coords = np.meshgrid(*ranges, indexing="ij")
    filter_size = coords[0].shape
    coords = np.stack(coords, axis=-1)

    new_shape = np.ones(input.ndim)
    new_shape = np.append(new_shape, -1).astype(int)
    sigmas = np.reshape(sigmas, new_shape)

    g = np.zeros(filter_size, dtype=np.complex)
    g[:] = np.exp(-0.5 * np.sum(np.divide(coords, sigmas) ** 2, axis=-1))

    g /= (2 * np.pi) ** (input.ndim / 2) * np.prod(sigmas)
    orientation = np.ones(input.ndim)
    for i, p in enumerate(phi):
        orientation[i + 1] = orientation[i] * np.sin(p)
        orientation[i] = orientation[i] * np.cos(p)
    orientation = np.flip(orientation)
    rotx = coords @ orientation
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    if isinstance(output, (type, np.dtype)):
        otype = output
    elif isinstance(output, str):
        otype = np.typeDict[output]
    else:
        otype = None

    output = ndi.convolve(
        input, weights=np.real(g), output=output, mode=mode, cval=cval
    )
    imag = ndi.convolve(input, weights=np.imag(g), output=otype, mode=mode, cval=cval)

    result = (output, imag)
    return result


def getLargestCC(segmentation: np.ndarray) -> np.ndarray:
    """Returns the largest connected component of a image.

    Arguments:
    segmentation : Segmentation data of image or volume.

    Returns:
    largeCC : Segmentation with only largest connected component.
    """

    check_type(segmentation, (list, np.ndarray))
    labels = label(segmentation)
    if labels.max() == 0:
        raise ValueError("No connected components!")  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def removeSmallCCs(segmentation: np.ndarray, size: Union[int, float]) -> np.ndarray:
    """Removes small connected components from an image.

    Parameters:
    segmentation : Segmentation data of image or volume.
    size : Maximum connected component size to remove.

    Returns:
    largeCCs : Segmentation with small connected components removed.
    """
    check_type(segmentation, (list, np.ndarray))
    check_type(size, numerical)

    labels = label(segmentation, return_num=False)
    counts = np.bincount(labels.flat)[1:]

    for v, count in enumerate(tqdm(counts, desc="looking for components to remove")):
        if count < size:
            labels[labels == v + 1] = 0

    largeCCs = labels != 0
    return largeCCs


def label_points(labels, points, res):
    """Adjust points so they fall on a foreground component of labels.

    Args:
        labels (array): labeled components, such as output from measure.label
        points (list): points to be adjusted
        res (list): voxel size

    Returns:
        [list]: adjusted points
        [list]: labels of adjusted points
    """
    point_labels = []
    nonzero_locs = np.argwhere(labels)
    for i, point in enumerate(points):
        too_big = [p >= l for p, l in zip(point, labels.shape)]
        if any(too_big) or labels[point[0], point[1], point[2]] == 0:
            dif = np.multiply(np.subtract(nonzero_locs, point), res)
            dists = np.linalg.norm(dif, axis=1)
            arg_min = np.argmin(dists)
            points[i] = nonzero_locs[arg_min, :]
        point = points[i]
        point_labels.append(labels[tuple(point)])
    return points, point_labels


def _get_chunked_args(soma_coords, labels, im_processed, chunk_size=[200, 200, 200]):
    """Splits large image data into smaller chunks so fragments can be generated in parallel.

    Args:
        soma_coords (list): list of voxel coordinates of somas
        labels (np.array): image segmentation
        im_processed (np.array): voxel-wise probability predictions for foreground
        chunk_size (list, optional): size of image chunks. Defaults to [200, 200, 200].

    Yields:
        dict: dictionary of arguments that depend on chunking
    """
    shp = labels.shape

    for x1 in np.arange(0, shp[0], chunk_size[0]):
        x2 = np.amin([x1 + chunk_size[0], shp[0]])
        for y1 in np.arange(0, shp[1], chunk_size[1]):
            y2 = np.amin([y1 + chunk_size[1], shp[1]])
            for z1 in np.arange(0, shp[2], chunk_size[2]):
                z2 = np.amin([z1 + chunk_size[2], shp[2]])
                soma_coords_new = []
                for soma_coord in soma_coords:
                    if (
                        np.less_equal([x1, y1, z1], soma_coord).all()
                        and np.less_equal(
                            soma_coord,
                            [x2, y2, z2],
                        ).all()
                    ):
                        soma_coords_new.append(np.subtract(soma_coord, [x1, y1, z1]))
                yield {
                    "soma_coords": soma_coords_new,
                    "labels": labels[x1:x2, y1:y2, z1:z2],
                    "im_processed": im_processed[x1:x2, y1:y2, z1:z2],
                }


def _merge_chunked_labels(labels, new_shape, chunk_size=[200, 200, 200]):
    """Merges the fragments of the chunked image. Assumes that chunking was done according to method in _get_chunked_args

    Args:
        labels (list): list of fragments generated by image chunks.
        new_shape (3-tuple of ints): size of stitched image
        chunk_size (list, optional): [description]. Defaults to [200, 200, 200].

    Returns:
        np.array: complete label image
    """
    new_labels = np.zeros(new_shape, dtype="int")
    idx = 0
    max = 0
    for x1 in np.arange(0, new_shape[0], chunk_size[0]):
        x2 = np.amin([x1 + chunk_size[0], new_shape[0]])
        for y1 in np.arange(0, new_shape[1], chunk_size[1]):
            y2 = np.amin([y1 + chunk_size[1], new_shape[1]])
            for z1 in np.arange(0, new_shape[2], chunk_size[2]):
                z2 = np.amin([z1 + chunk_size[2], new_shape[2]])

                lab = labels[idx]
                lab[lab > 0] += max
                new_labels[x1:x2, y1:y2, z1:z2] = lab

                max = np.amax(lab)
                idx += 1
    return new_labels


def compute_frags(
    soma_coords, labels, im_processed, threshold, res, chunk_size=None, ncpu=2
):
    """Preprocesses a neuron image segmentation by splitting up non-soma components into 5 micron segments.

    Args:
        soma_coords (list): list of voxel coordinates of somas
        labels (np.array): image segmentation
        im_processed (np.array): voxel-wise probability predictions for foreground
        threshold (float): threshold used to segment probability predictions into mask
        res (list): voxel size in image
        chunk_size (list): size of image chunks
        ncpu (int): number of cpus to use in parallel mode

    Returns:
        np.array: new image segmentation - different numbers indicate different fragments, 0 is background
    """
    og_shape = labels.shape
    if chunk_size is None:
        new_labels = split_frags(soma_coords, labels, im_processed, threshold, res)
    else:
        args = _get_chunked_args(
            soma_coords, labels, im_processed, chunk_size=chunk_size
        )
        inputs = [
            (arg["soma_coords"], arg["labels"], arg["im_processed"], threshold, res)
            for arg in args
        ]
        with mp.Pool(ncpu) as pool:
            new_labelss = pool.starmap(split_frags, inputs)

        new_labels = _merge_chunked_labels(new_labelss, og_shape, chunk_size=chunk_size)
    return new_labels


def split_frags(soma_coords, labels, im_processed, threshold, res, verbose=True):
    """Preprocesses a single image chunk by splitting up non-soma components into 5 micron segments

    Args:
        soma_coords (list): list of voxel coordinates of somas
        labels (np.array): image segmentation
        im_processed (np.array): voxel-wise probability predictions for foreground
        threshold (float): threshold used to segment probability predictions into mask
        res (list): voxel size in image

    Returns:
        np.array: new image segmentation - different numbers indicate different fragments, 0 is background
    """
    radius_states = 7
    image_iterative, states, comp_to_states, new_soma_masks = remove_somas(
        soma_coords, labels, im_processed, res
    )

    mask = labels > 0
    mask2 = removeSmallCCs(mask, 25)
    image_iterative[mask & (~mask2)] = 0

    states, comp_to_states = split_frags_place_points(
        image_iterative,
        labels,
        radius_states,
        res,
        threshold,
        states,
        comp_to_states,
    )

    new_labels = split_frags_split_comps(labels, new_soma_masks, states, comp_to_states)

    new_labels = split_frags_split_fractured_components(new_labels)

    props = regionprops(new_labels)
    for label, prop in enumerate(
        tqdm(props, desc="remove small fragments", disable=not verbose)
    ):
        if prop.area < 15:
            new_labels[new_labels == prop.label] = 0

    new_labels = rename_states_consecutively(new_labels)

    return new_labels


def remove_somas(soma_coords, labels, im_processed, res, verbose=True):
    """Helper function of split_frags. Removes area around somas.

    Args:
        soma_coords (list): list of voxel coordinates of somas
        labels (np.array): image segmentation
        im_processed (np.array): voxel-wise probability predictions for foreground
        res (list): voxel size in image

    Returns:
        np.array: probability predictions, with the soma regions masked
        list: coordinates of the points
        dictionary: map from component in labels, to set of points that were placed there
        list: masks of the different somas
    """
    states = []
    comp_to_states = {}
    # probability image, with all soma regions set to 0
    image_iterative = np.copy(im_processed)
    # list of soma region masks
    new_soma_masks = []

    for soma_pt in tqdm(soma_coords, desc="removing somas", disable=not verbose):
        _, end_lbls = label_points(labels, [soma_pt], res)
        soma_lbl = end_lbls[0]
        soma_mask = labels == soma_lbl

        states.append(np.array(soma_pt))
        comp = labels[soma_pt[0], soma_pt[1], soma_pt[2]]
        comp_to_states[comp] = [len(states) - 1]

        # soma component is all the voxels of that component within 12 microns of the soma point
        dist = np.ones_like(image_iterative)
        dist[soma_pt[0], soma_pt[1], soma_pt[2]] = 0
        dt = ndi.morphology.distance_transform_edt(dist, sampling=[0.3, 0.3, 1])
        sphere = dt < 15
        new_soma_mask = np.logical_and(soma_mask, sphere)

        image_iterative[new_soma_mask] = 0
        new_soma_masks.append(new_soma_mask)

    return image_iterative, states, comp_to_states, new_soma_masks


def split_frags_place_points(
    image_iterative,
    labels,
    radius_states,
    res,
    threshold,
    states,
    comp_to_states,
    verbose=True,
):
    """Helper function of split_frags. Places points on high probability voxels while keeping the points a certain distance apart from each other.

    Args:
        image_iterative (np.array): probability predictions, with the soma regions masked
        labels (np.array): image segmentation
        radius_states (float): distance constraint between points
        res (list): voxel size in image
        threshold (float): threshold used to segment probability predictions into mask
        states (list): coordinates of the points
        comp_to_states (dictionary): map from component in labels, to set of points that were placed there

    Returns:
        list: coordinates of the points
        dictionary: map from component in labels, to set of points that were placed there
    """
    top_ind = np.unravel_index(
        np.argmax(image_iterative, axis=None), image_iterative.shape
    )
    top = image_iterative[top_ind[0], top_ind[1], top_ind[2]]

    radius_vox = np.divide(radius_states, res).astype(int)

    prev_tot = np.sum(image_iterative > threshold)

    with tqdm(total=prev_tot, desc="Adding points...", disable=not verbose) as pbar:
        while top > threshold:
            states.append(top_ind)

            comp = labels[top_ind[0], top_ind[1], top_ind[2]]
            if comp in comp_to_states.keys():
                lst = comp_to_states[comp]
                lst.append(len(states) - 1)
                comp_to_states[comp] = lst
            else:
                comp_to_states[comp] = [len(states) - 1]

            l_bd = [
                np.amax([0, top_ind[0] - radius_vox[0]]),
                np.amax([0, top_ind[1] - radius_vox[1]]),
                np.amax([0, top_ind[2] - radius_vox[2]]),
            ]
            u_bd = [
                np.amin([image_iterative.shape[0], top_ind[0] + radius_vox[0]]),
                np.amin([image_iterative.shape[1], top_ind[1] + radius_vox[1]]),
                np.amin([image_iterative.shape[2], top_ind[2] + radius_vox[2]]),
            ]
            image_iterative[l_bd[0] : u_bd[0], l_bd[1] : u_bd[1], l_bd[2] : u_bd[2]] = 0

            top_ind = np.unravel_index(
                np.argmax(image_iterative, axis=None), image_iterative.shape
            )
            top = image_iterative[top_ind[0], top_ind[1], top_ind[2]]

            tot = np.sum(image_iterative > threshold)
            pbar.update(prev_tot - tot)
            prev_tot = tot

    return states, comp_to_states


def split_frags_split_comps(
    labels, new_soma_masks, states, comp_to_states, verbose=True
):
    """Helper function of split_frags. Splits the components according to the points that were placed by split_frags_place_points.

    Args:
        labels (np.array): image segmentation
        new_soma_masks ([type]): [description]
        states (list): coordinates of the points
        comp_to_states (dictionary): map from component in labels, to set of points that were placed there

    Returns:
        np.array: new image segmentation - different numbers indicate different fragments, 0 is background
    """
    labels_split = np.copy(labels)

    next_lbl = np.amax(labels) + 1
    for comp in tqdm(
        comp_to_states.keys(), desc="Splitting Fragments", disable=not verbose
    ):
        comp_states = comp_to_states[comp]
        if len(comp_states) > 1:
            state_coords = []
            for state in comp_states:
                state_coords.append(states[state])
            state_coords = np.stack(state_coords)
            comp_coords = np.argwhere(labels == comp)
            amin, _ = pairwise_distances_argmin_min(comp_coords, state_coords)

            for s, state in enumerate(np.unique(amin)):
                if s > 0:
                    coords = comp_coords[amin == state]
                    labels_split[coords[:, 0], coords[:, 1], coords[:, 2]] = next_lbl
                    next_lbl += 1
    mx = np.amax(labels_split)

    for i, new_soma_mask in enumerate(new_soma_masks):
        labels_split[new_soma_mask] = mx + 1 + i
    new_labels = labels_split
    return new_labels


def split_frags_split_fractured_components(new_labels, verbose=True):
    """Helper function of split_frags. Some fragments from split_frags_split_comps may not be connected so this function separates those.

    Args:
        new_labels (np.array): new image segmentation - different numbers indicate different fragments, 0 is background

    Returns:
        np.array: new image segmentation - different numbers indicate different fragments, 0 is background
    """
    props = regionprops(new_labels)
    new_lbl = np.amax(new_labels) + 1
    for prop in tqdm(props, desc="Split fractured components", disable=not verbose):
        bbox = prop["bbox"]
        lbl = prop["label"]
        cutout = new_labels[bbox[0] : bbox[3], bbox[1] : bbox[4], bbox[2] : bbox[5]]
        mask = cutout == lbl
        lbl_labels = label(mask)
        for lbl_label in np.unique(lbl_labels):
            if lbl_label not in [0, 1]:
                cutout[lbl_labels == lbl_label] = new_lbl
                new_lbl += 1

    return new_labels


def rename_states_consecutively(new_labels):
    """Helper function of split_frags. Relabel components in image segmentation so the unique values are consecutive.

    Args:
        new_labels (np.array): new image segmentation - different numbers indicate different fragments, 0 is background

    Returns:
        np.array: new image segmentation - different numbers indicate different fragments, 0 is background
    """
    vals = np.unique(new_labels)
    vals = np.delete(vals, 0)
    vals = np.append(vals, [0])
    new_vals = np.arange(1, len(vals))
    new_vals = np.append(new_vals, [0])

    data = np.reshape(np.copy(new_labels), (new_labels.size,))
    sort_idx = np.argsort(vals)
    idx = np.searchsorted(vals, data, sorter=sort_idx)
    out = new_vals[sort_idx][idx]
    new_labels = np.reshape(out, new_labels.shape)

    return new_labels
