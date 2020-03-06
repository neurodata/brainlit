import numpy as np
from skimage.measure import label
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from itertools import product


def gabor_filter(input, sigma, phi, frequency, offset=0.0, 
                output=None, mode="reflect", cval=0.0,
                truncate=4.0,):
    """Multidimensional Gabor filter. A gabor filter
    is an elementwise product between a Gaussian 
    and a complex exponential.

    Parameters
    ----------
    %(input)s
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
        Frequency of the complex exponential. Units are revolutions/voxels. <- other one is pixels
    offset : scalar
        Phase shift of the complex exponential. Units are radians.

    %(output)s
    %(mode_multiple)s
    %(cval)s
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
    >>> from scipy.ndimage import gabor_filter
    >>> a = np.arange(50, step=2).reshape((5,5))
    >>> a
    array([[ 0,  2,  4,  6,  8],
           [10, 12, 14, 16, 18],
           [20, 22, 24, 26, 28],
           [30, 32, 34, 36, 38],
           [40, 42, 44, 46, 48]])
    >>> gabor_filter(a, sigma=1, phi=[0.0], frequency=0.1)
    array([[ ? ]])

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
    
    g /= (2 * np.pi)** (input.ndim / 2) * np.prod(sigmas)
    orientation = np.ones(input.ndim)
    for i, p in enumerate(phi):
        orientation[i + 1] = orientation[i] * np.sin(p)
        orientation[i] = orientation[i] * np.cos(p)
    orientation = np.flip(orientation)
    rotx = coords @ orientation
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    real = ndi.convolve(input, weights=np.real(g), output=output, mode=mode, cval=cval)
    imaginary = ndi.convolve(input, weights=np.imag(g), output=output, mode=mode, cval=cval)

    output = (real, imaginary)
    return output


def getLargestCC(segmentation):
    """Returns the largest connected component of a image

    Parameters
    -------
    segmentation : array-like
        segmentation data of image or volume

    Returns
    -------
    largeCC : array-like
        segmentation with only largest connected component

    """

    labels = label(segmentation)
    if labels.max() == 0:
        raise ValueError("No connected components!")  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def removeSmallCCs(segmentation, size):
    """Removes small connected components from an image

    Parameters
    -------
    segmentation : array-like
        segmentation data of image or volume

    size : scalar
        maximize connected component size to remove

    Returns
    -------
    largeCCs : array-like
        segmentation with small connected components removed

    """
    labels = label(segmentation, return_num=False)

    if labels.max() == 0:
        raise ValueError("No connected components!")
    counts = np.bincount(labels.flat)[1:]

    for v, count in enumerate(counts):
        if count < size:
            labels[labels == v + 1] = 0

    largeCCs = labels != 0
    return largeCCs