import numpy as np
from skimage.measure import label
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from itertools import product


def otsu_segment(data):
    """Otsu binary segmentation
    
    Arguments:
        data {numpy array of any shape/size} -- data to be segmented

    Returns:
        mask -- binary segmentation
    """
    unq = np.unique(data)
    best_cost = np.inf
    threshold = np.inf
    for val in unq:

        n0 = np.count_nonzero(data <= val)
        var0 = np.var(data[data <= val])

        n1 = np.count_nonzero(data > val)
        if n1 > 0:
            var1 = np.var(data[data > val])

        cost = n0 * var0 + n1 * var1
        if cost < best_cost:
            best_cost = cost
            threshold = val

    mask = np.greater(data, threshold)
    mask = mask.astype(int)
    return mask, threshold


def gabor_kernel_3d(
    frequency,
    theta=0,
    phi=0,
    sigma_x=None,
    sigma_y=None,
    sigma_z=None,
    n_stds=3,
    offset=0,
):
    # As in skimage, we interpret the inputs as sigmas along the principal axes of the Gaussian ellipse
    # Thus, we need trigonometry to project the principal axes along the standard basis
    x0 = np.ceil(
        max(
            np.abs(n_stds * sigma_x * np.cos(theta) * np.cos(phi)),
            np.abs(n_stds * sigma_y * np.sin(phi)),
            np.abs(n_stds * sigma_z * np.sin(theta) * np.cos(phi)),
            1,
        )
    )
    y0 = np.ceil(
        max(
            np.abs(n_stds * sigma_x * np.cos(theta) * np.sin(phi)),
            np.abs(n_stds * sigma_y * np.cos(phi)),
            np.abs(n_stds * sigma_z * np.sin(theta) * np.sin(phi)),
            1,
        )
    )
    z0 = np.ceil(
        max(
            np.abs(n_stds * sigma_x * np.sin(theta)),
            np.abs(n_stds * sigma_z * np.cos(theta)),
            1,
        )
    )

    x, y, z = np.mgrid[-x0 : x0 + 1, -y0 : y0 + 1, -z0 : z0 + 1]

    # the rot coordinates come from the inverse rotations by theta and phi
    rotx = (
        x * np.cos(theta) * np.cos(phi)
        + y * np.cos(theta) * np.sin(phi)
        - z * np.sin(theta)
    )
    roty = -x * np.sin(phi) + y * np.cos(phi)
    rotz = (
        x * np.sin(theta) * np.cos(phi)
        + y * np.sin(theta) * np.cos(phi)
        + z * np.cos(theta)
    )

    g = np.zeros(x.shape, dtype=np.complex)
    g[:] = np.exp(
        -0.5
        * (
            rotx ** 2 / sigma_x ** 2
            + roty ** 2 / sigma_y ** 2
            + rotz ** 2 / sigma_z ** 2
        )
    )
    g /= 2 * np.pi * sigma_x * sigma_y * sigma_z
    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    return g


def gabor_filter(
    input,
    sigma,
    phi,
    frequency,
    offset=0.0,
    output=None,
    mode="reflect",
    cval=0.0,
    truncate=4.0,
):
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
        Frequency of the complex exponential. Units are revolutions/voxels.
    offset : scalar
        Phase shift of the complex exponential. Units are voxels.

    %(output)s
    %(mode_multiple)s
    %(cval)s
    truncate : float
        Truncate the filter at this many standard deviations.
        Default is 4.0.

    Returns
    -------
    gabor_filter : ndarray
        Returned array of same shape as `input`.

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
    >>> ax2.imshow(result)
    >>> plt.show()
    """
    input = np.asarray(input)
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

    g /= 2 * np.pi * np.prod(sigmas)

    orientation = np.ones(input.ndim)
    for i, p in enumerate(phi):
        orientation[i + 1] = orientation[i] * np.sin(p)
        orientation[i] = orientation[i] * np.cos(p)

    rotx = coords @ orientation

    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))

    output = ndi.convolve(input, weights=g, output=output, mode=mode, cval=cval)

    return output


def gabor_kernel_nd(sigma, phi, frequency, offset=0.0, truncate=3.0):
    dim = len(phi) + 1
    sigmas = ndi._ni_support._normalize_sequence(sigma, dim)
    phi = ndi._ni_support._normalize_sequence(phi, dim - 1)

    limits = [np.ceil(truncate * sigma).astype(int) for sigma in sigmas]
    ranges = [range(-limit, limit + 1) for limit in limits]
    coords = np.meshgrid(*ranges, indexing="ij")
    filter_size = coords[0].shape
    coords = np.stack(coords, axis=-1)

    new_shape = np.ones(dim)
    new_shape = np.append(new_shape, -1).astype(int)
    sigmas = np.reshape(sigmas, new_shape)

    g = np.zeros(filter_size, dtype=np.complex)
    g[:] = np.exp(-0.5 * np.sum(np.divide(coords, sigmas) ** 2, axis=-1))

    g /= 2 * np.pi * np.prod(sigmas)

    orientation = np.ones(dim)
    for i, p in enumerate(phi):
        orientation[i + 1] = orientation[i] * np.sin(p)
        orientation[i] = orientation[i] * np.cos(p)

    rotx = coords @ orientation

    g *= np.exp(1j * (2 * np.pi * frequency * rotx + offset))
    return g


def getLargestCC(segmentation):
    labels = label(segmentation)
    assert labels.max() != 0  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def removeSmallCCs(segmentation, size):
    labels = label(segmentation, return_num=False)

    assert labels.max() != 0  # assume at least 1 CC
    counts = np.bincount(labels.flat)[1:]

    for v, count in enumerate(counts):
        if count < size:
            labels[labels == v + 1] = 0

    largeCCs = labels != 0
    return largeCCs


def getKernels(neighborhood=None):
    if neighborhood is not None:
        return getNeighborhoodKernels(neighborhood)
    # ******Create filters
    names = []
    kernels = []
    # Normal filters
    r = 6
    sz = 2 * r + 1

    dirac = np.zeros((sz, sz, sz))
    dirac[r, r, r] = 1
    kernels.append(dirac)
    names.append("0 dirac")

    sigs = [0.5, 1, 2]

    for s, sigma in enumerate(sigs):
        gaussian = ndi.gaussian_filter(dirac, (sigma, sigma, 0.3 * sigma))
        kernels.append(gaussian)
        name = str(s + 1) + " gaussian" + str(sigma)
        names.append(name)

    # Gabor filters
    freqs = [0.05, 0.1]
    sigs = [[2, 2, 0.6], [4, 4, 1.2]]
    offsets = [-np.pi / 2, 0, np.pi / 2]
    angles = [[0, 0], [np.pi / 2, 0]]

    for p, params in enumerate(product(freqs, sigs, offsets, angles)):
        freq, sig, offset, angle = params
        filter = gabor_kernel_nd(sigma=sig, phi=angle, frequency=freq, offset=offset)
        kernels.append(np.real(filter))
        name = (
            str(p + 4)
            + " gabor_"
            + str(freq)
            + "_"
            + str(sig)
            + "_%1.2f" % offset
            + "_"
            + str(angle)
        )
        names.append(name)

    # to deal with anisotropy, we change the frequency
    # when the periodic function points in the x3 direction
    angles = [[np.pi / 2, np.pi / 2]]
    freqs = [0.167, 0.333]
    for p, params in enumerate(product(freqs, sigs, offsets, angles)):
        freq, sig, offset, angle = params
        filter = gabor_kernel_nd(sigma=sig, phi=angle, frequency=freq, offset=offset)
        kernels.append(np.real(filter))
        name = (
            str(p + 28)
            + " gabor_"
            + str(freq)
            + "_"
            + str(sig)
            + "_%1.2f" % offset
            + "_"
            + str(angle)
        )
        names.append(name)

    return kernels, names


def getNeighborhoodKernels(radii=[1, 1, 1]):
    radii = np.array(radii)
    shp = radii * 2 + 1
    sz = np.product(shp)
    kernels = []
    for i in range(sz):
        ker = np.zeros((sz))
        ker[i] = 1
        ker = np.reshape(ker, shp)
        # because convolving will flip it back over
        ker = np.flip(ker)
        kernels.append(ker)

    return kernels, None
