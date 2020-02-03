import numpy as np
from pathlib import Path
from skimage import io


def get_coord_params(path):
    """Get basic parameters of the image data so spatial coordinates can be converted to voxel coordinates

        Arguments:
            path {string} -- path to highest level of octree

        Returns:
            origin {3-array} -- spatial location of origin in microns
            size {3-array} -- voxel dimensions at finest level
            spacing {3-array} -- spacing at finest resolution level in microns
            nl {int} -- number of levels in octree
    """
    parent_path = Path(path)
    transform_path = parent_path / "transform.txt"

    # the expected order of parameters in transform.txt
    content = ["ox", "oy", "oz", "sx", "sy", "sz", "nl"]
    values = []

    with transform_path.open() as t:
        lines = t.readlines()
        for l, line in enumerate(lines):
            words = line.split()
            idx = words[0].find(content[l])
            if idx == -1:
                raise ValueError("transform.txt not formatted as expected")

            # all fields are floats except level number parameter
            if l == 6:
                values.append(int(words[1]))
            else:
                values.append(float(words[1]) / 1000)  # convert nm to um

    nl = values[6]

    top_spacing = np.array([values[3], values[4], values[5]])
    num_voxels = np.power(2, nl - 1)
    spacing = np.divide(top_spacing, num_voxels)

    origin = np.array([values[0], values[1], values[2]])
    # The following modification is according to an email from Adam Taylor on 9/18/19
    # This origin shift is important when we decode the spatial locations of swc files
    origin = np.multiply(spacing, np.floor(np.divide(origin, spacing)))

    img_path = parent_path / "default.0.tif"

    im = io.imread(str(img_path))
    dims = im.shape

    size = np.array([dims[2], dims[1], dims[0]]) * np.power(2, nl - 1)
    size = size.astype(int)

    top_spacing = np.array([values[3], values[4], values[5]])

    return origin, size, spacing, nl


def space_to_voxel(spatial_coord, origin, spacing):
    """Convert spatial location to voxel coordinates

    Arguments:
        spatial_coord {3-array} -- spatial coordinates in microns
        origin {3-array} -- spatial coordinates of origin in microns
        spacing {3-array} -- size of each voxel in microns

    Returns:
        voxel_coord {3-array of ints} -- voxel coordinate
    """

    voxel_coord = np.round(np.divide(spatial_coord - origin, spacing))
    voxel_coord = voxel_coord.astype(int)
    return voxel_coord


def voxel_to_space(vox, origin, spacing):
    """Convert voxel coordinate to spatial location

    Arguments:
        vox {3-array of ints} -- voxel coordinate
        origin {3-array} -- spatial coordinates of origin in microns
        spacing {3-array} -- size of each voxel in microns
    Returns:
         spatial_coord {3-array} -- spatial coordinates in microns
    """
    spatial_coord = np.multiply(vox, spacing) + origin
    return spatial_coord


def voxel_path(vox, tree_specifics):
    """output the image path that contains the voxel and its coordinate in that file

    Arguments:
        vox {3-array of ints} -- voxel coordinates [0, sz-1]
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}

    Returns:
        img_path -- path to the image that stores that voxel
        vox_cur -- coordinate of the voxel in that image
    """
    path, sz, nl, channel = tree_specifics
    path_cur = Path(path)
    sz_cur = sz
    vox_cur = vox
    for l in range(0, nl - 1):
        sz_cur = np.divide(sz_cur, 2).astype(int)
        split = np.greater(vox_cur, sz_cur - 1)
        leaf = np.dot(split, np.array([1, 2, 4])) + 1
        path_cur = path_cur / str(leaf)
        vox_cur = vox_cur - np.multiply(sz_cur, split)
    file = "default." + str(channel) + ".tif"
    img_path = path_cur / file

    return img_path, vox_cur


def find_voxel(vox, tree_specifics):
    """output the pixel at a voxel location

    Arguments:
        vox {3-array of ints} -- voxel coordinates [0, sz-1]
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}

    Returns:
        pixel - voxel value at that location, 0 if undefined
    """
    img_path, vox_cur = voxel_path(vox, tree_specifics)
    try:
        im = io.imread(str(img_path))
        pixel = im[vox_cur[2], vox_cur[1], vox_cur[0]]
    except FileNotFoundError:
        pixel = np.nan
    return pixel


def divide_voxel_bounds(start, end, tree_specifics):
    """If the corner coordinates of the desired subvolume span multiple tifs, then this function splits
    each dimension at the limits of the tifs

    Arguments:
        start {3-array of ints} -- start corner of rectangle
        end {3-array of ints} -- end corner of rectangle
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}
    Returns:
        segments {3-array} -- each element is a list of segments along the axis that lines up with the borders of the tif volumes
    """
    _, start_coord = voxel_path(start, tree_specifics)
    _, end_coord = voxel_path(end, tree_specifics)
    sz = tree_specifics[1]
    sz_single = np.divide(sz, 2 ** 6)
    segments = []

    for dim in range(len(sz)):
        coords = []
        left = start[dim]
        right = (np.floor_divide(left, sz_single[dim]) + 1) * sz_single[dim] - 1
        right = right.astype(int)
        while right < end[dim]:
            coords.append([left, right])
            left = right + 1
            right = (np.floor_divide(left, sz_single[dim]) + 1) * sz_single[dim] - 1
            right = right.astype(int)
        coords.append([left, end[dim]])
        segments.append(coords)
    return segments


def stitch_images(segments, tree_specifics):
    """Stitch together image subvolumes

    Arguments:
        segments {3 list} -- segment locations as outputted by divide_voxel_bounds()
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}

    Returns:
        volume -- image subvolume
    """
    if len(segments[0]) > 1:
        segments1 = segments[:]
        segments1[0] = [segments1[0][0]]

        segments2 = segments[:]
        segments2[0] = segments2[0][1:]

        left = stitch_images(segments1, tree_specifics)
        right = stitch_images(segments2, tree_specifics)

        return np.concatenate((left, right), axis=0)
    elif len(segments[1]) > 1:
        segments1 = segments[:]
        segments1[1] = [segments1[1][0]]

        segments2 = segments[:]
        segments2[1] = segments2[1][1:]

        left = stitch_images(segments1, tree_specifics)
        right = stitch_images(segments2, tree_specifics)

        return np.concatenate((left, right), axis=1)
    elif len(segments[2]) > 2:
        segments1 = segments[:]
        segments1[2] = [segments1[2][0]]

        segments2 = segments[:]
        segments2[2] = segments2[2][1:]

        left = stitch_images(segments1, tree_specifics)
        right = stitch_images(segments2, tree_specifics)

        return np.concatenate((left, right), axis=2)
    elif len(segments[2]) == 2:
        start1 = [segments[0][0][0], segments[1][0][0], segments[2][0][0]]
        end1 = [segments[0][0][1], segments[1][0][1], segments[2][0][1]]
        left = get_intrarectangle_voxel(start1, end1, tree_specifics)

        start2 = [segments[0][0][0], segments[1][0][0], segments[2][1][0]]
        end2 = [segments[0][0][1], segments[1][0][1], segments[2][1][1]]
        right = get_intrarectangle_voxel(start2, end2, tree_specifics)

        return np.concatenate((left, right), axis=2)
    else:
        start1 = [segments[0][0][0], segments[1][0][0], segments[2][0][0]]
        end1 = [segments[0][0][1], segments[1][0][1], segments[2][0][1]]
        left = get_intrarectangle_voxel(start1, end1, tree_specifics)
        return left


def get_intrarectangle_voxel(start, end, tree_specifics):
    """Get a subvolume of image that is contained within a single tif image

    Arguments:
        start {3 array of ints} -- lower corner of rectangle
        end {3 array of ints} -- upper corner of rectangle (to be included)
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}
    """
    start_path, start_coord = voxel_path(start, tree_specifics)
    end_path, end_coord = voxel_path(end, tree_specifics)
    end_coord = end_coord + 1

    if start_path != end_path:
        raise ValueError(
            "Call to get_intrarectangle_voxel for coordinates that are in different files"
        )
    try:
        im = io.imread(str(start_path))
        volume = im[
            start_coord[2] : end_coord[2],
            start_coord[1] : end_coord[1],
            start_coord[0] : end_coord[0],
        ]
        volume = np.swapaxes(volume, 0, 2)
    except FileNotFoundError:
        volume = np.nan * np.zeros(end_coord - start_coord)
    return volume


def get_interrectangle_voxel(start, end, tree_specifics):
    """retrieve a 3d rectangle of image data given the voxel coordinates of the two corners

    Arguments:
        start {3-array of ints} -- start corner of rectangle
        end {3-array of ints} -- end corner of rectangle
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}

    Returns:
        volume -- image subvolume
    """
    segments = divide_voxel_bounds(start, end, tree_specifics)
    volume = stitch_images(segments, tree_specifics)
    return volume


def get_rectangle_spatial(path, coord, radii, channel, coord_params=None):
    """Finds a cube of image data around a specified coordinate

        Arguments:
            path {string} -- path to highest level of octree
            coord {3-array} -- spatial coordinates
            radii {3-array of ints} -- number of voxels around the coordinate to include, in the order x,y,z
            channel {0 or 1} -- desired image channel
            coord_params {list} -- parameters about the image volume, formatted as an output of get_coord_params
        Returns:
            rectangle -- image subvolume

    """
    if coord_params == None:
        [o, sz, sp, nl] = get_coord_params(path)
    else:
        [o, sz, sp, nl] = coord_params
    voxel_coord = space_to_voxel(coord, o, sp)

    start_coord = voxel_coord - radii
    end_coord = voxel_coord + radii
    tree_specifics = [path, sz, nl, channel]
    rectangle = get_interrectangle_voxel(start_coord, end_coord, tree_specifics)

    return rectangle


def path_to_leftcoord(path, tree_specifics):
    """Given a path to a tif file, output the voxel coordinate of its first entry

    Arguments:
        path {string or Path object} -- tif file path
        tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}

    Returns:
        lef {3-array} -- voxel coordinate of left corner (ie smallest coordinate) in a given tif image
    """
    _, sz, nl, _ = tree_specifics
    sz_cur = sz
    left = np.array([0, 0, 0])

    if isinstance(path, str):
        path = Path(path)
    string = str(path)
    dirs = string.split("/")
    dirs = dirs[-nl:-1]

    for dir in dirs:
        d = int(dir)
        sz_cur = np.divide(sz_cur, 2).astype(int)
        binary = bin(d - 1)[2:]
        while len(binary) < 3:
            binary = "0" + binary
        binary_list = list(int(binary[-(i + 1)]) for i in range(len(binary)))
        binary_list = np.array(binary_list)
        left = left + np.multiply(sz_cur, binary_list)
    return left


def get_neighborhood(im, idx, rad=[1, 1, 1]):
    """Get an image neighborhood around a location. Uses edge padding
    
    Arguments:
        im {ndarray} -- image
        idx {array like} -- coordinates of center
    
    Keyword Arguments:
        rad {list} -- radius of the neighborhood in each direction (default: {[1,1,1]})
    
    Returns:
        ndarray -- neighborhood of the image
    """
    if type(idx) is not np.ndarray:
        idx = np.array(idx)

    pad_width = [[r, r] for r in rad]
    im_pad = np.pad(im, pad_width, mode="edge")

    start = idx + rad
    end = idx + rad + rad + 1

    coords = list(zip(start, end))
    slices = tuple(slice(coord[0], coord[1]) for coord in coords)

    return im_pad[slices]
