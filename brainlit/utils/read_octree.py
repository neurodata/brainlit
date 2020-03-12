import numpy as np
from pathlib import Path
from skimage import io
import cv2


class octree(object):
    def __init__(self, root_dir):
        """Create octree object based on the root directory

            Arguments:
                root_dir {string} -- path to highest level of octree

            Variables:
                root_dir {string} -- path to highest level of octree
                origin {3-array} -- spatial location of origin in microns
                size {3-array} -- voxel dimension of entire octree volume
                top_spacing {3-array} -- spacing at highest level in microns
                nl {int} -- number of levels in octree
                bot_spacing {3-array} -- spacing at lowest level in microns
                tif_dim {3-array} -- (x,y,z) voxel dimension of each tif image, which is same for all levels
                size_spatial {3-array} -- (x,y,z) spatial dimension of entire octree volume in um
        """
        self.root_dir = root_dir

        # get all octree coordinate parameters
        self.origin, self.size, self.top_spacing, self.nl = self.get_coord_params()
        self.bot_spacing = self.level_to_spacing(self.nl)

        # get voxel dimension of tif image
        im = io.imread(str(Path(root_dir) / "default.0.tif"))
        shp = im.shape
        self.tif_dim = np.array([shp[2], shp[1], shp[0]])
        self.size_spatial = self.tif_dim * self.top_spacing

        print("number of levels of octree:", str(self.nl))
        print("origin in spatial coords:", str(self.origin))
        print(
            "spatial dimension of entire octree volume in um:", str(self.size_spatial)
        )
        print("voxel dimension of entire octree volume :", str(self.size))
        print("spacing at highest level in um:", str(self.top_spacing))
        print("spacing at lowest level in um:", str(self.bot_spacing))
        print("tif image shape (x,y,z):", str(self.tif_dim))

    def get_coord_params(self):
        """Get basic parameters of the image data so spatial coordinates can be converted to voxel coordinates

            Arguments:
                root_dir {string} -- path to highest level of octree

            Returns:
                origin {3-array} -- spatial location of origin in microns
                size {3-array} -- voxel dimensions at lowest level
                top_spacing {3-array} -- spacing at highest level in microns
                nl {int} -- number of levels in octree
        """

        parent_path = Path(self.root_dir)
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

        origin = np.array([values[0], values[1], values[2]])
        nl = values[6]

        img_path = parent_path / "default.0.tif"

        im = io.imread(str(img_path))
        dims = im.shape

        size = np.array([dims[2], dims[1], dims[0]]) * np.power(2, nl - 1)
        size = size.astype(int)

        top_spacing = np.array([values[3], values[4], values[5]])

        return origin, size, top_spacing, nl

    def level_to_spacing(self, level):
        """Find the spacing of each voxel at a paricular level

        Arguments:
            level {int} -- level of the octree

        Returns:
            spacing {3-array} -- size of each voxel in microns
        """
        num_voxels = np.power(2, level - 1)
        spacing = np.divide(self.top_spacing, num_voxels)
        return spacing
        ## TODO: calculate spacings of all levels and store as class variable

    def space_to_voxel(self, spatial_coord, level=None):
        """Convert spatial location to voxel coordinates relative to the level

        Arguments:
            spatial_coord {3-array} -- spatial coordinates in microns
            level {int} -- level in octree; default is maximum level, nl

        Returns:
            voxel_coord {3-array of ints} -- voxel coordinate relative to level
        """
        # default value for level
        if level == None:
            level = self.nl

        spacing = self.level_to_spacing(level)
        voxel_coord = np.round(np.divide(spatial_coord - self.origin, spacing))
        voxel_coord = voxel_coord.astype(int)
        return voxel_coord

    def voxel_to_space(self, vox, level=None):
        """Convert voxel coordinate to spatial location
    
        Arguments:
            vox {3-array of ints} -- voxel coordinates relative to entire octree volume at lowest level
            level {int} -- level in octree; default is maximum level, nl
        Returns:
             spatial_coord {3-array} -- spatial coordinates in microns
        """
        # default value for level
        if level == None:
            level = self.nl

        spacing = self.level_to_spacing(level)
        spatial_coord = np.multiply(vox, spacing) + self.origin
        return spatial_coord

    def voxel_path(self, vox, channel=0):
        """output the lowest level octant image path that contains the voxel and its coordinate in that file

        Arguments:
            vox {3-array of ints} -- voxel coordinates relative to entire octree volume at lowest level
            tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}
            channel {0 or 1} -- desired image channel; default is 0

        Returns:
            img_path -- path to the image that stores that voxel
            vox_cur -- coordinate of the voxel in that octant image
        """
        path_cur = Path(self.root_dir)
        sz_cur = self.size
        vox_cur = vox
        for l in range(0, self.nl - 1):
            sz_cur = np.divide(sz_cur, 2).astype(int)
            split = np.greater(
                vox_cur, sz_cur - 1
            )  # get lower/upper half split boolean for 3 axis
            node = np.dot(split, np.array([1, 2, 4])) + 1  # get folder name
            path_cur = path_cur / str(node)  # append folder name to path
            vox_cur = vox_cur - np.multiply(sz_cur, split)
        file = "default." + str(channel) + ".tif"
        img_path = path_cur / file

        return img_path, vox_cur

    def find_voxel(self, vox):
        """output the pixel at a voxel location

        Arguments:
            vox {3-array of ints} -- voxel coordinates relative to entire octree volume at lowest level

        Returns:
            pixel - voxel value at that location, 0 if undefined
        """
        img_path, vox_cur = self.voxel_path(vox)
        try:
            im = io.imread(str(img_path))  # reads octant as (z,y,x)
            pixel = im[vox_cur[2], vox_cur[1], vox_cur[0]]
        except FileNotFoundError:
            pixel = np.nan
        return pixel

    def divide_voxel_bounds(self, start, end):
        """If the corner coordinates of the desired subvolume span multiple lowest level octants (tif files), then this function splits
        each dimension at the limits of the tifs

        Arguments:
            start {3-array of ints} -- start corner of rectangle
            end {3-array of ints} -- end corner of rectangle

        Returns:
            segments {list of length 3} -- each element is a list of segments along the axis that lines up with the borders of the tif volumes
        """
        _, start_coord = self.voxel_path(start)
        _, end_coord = self.voxel_path(end)
        sz = self.size
        sz_single = np.divide(sz, 2 ** (self.nl - 1))
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

    def get_intrarectangle_voxel(self, start, end, channel):
        """Get a subvolume of image that is contained within a single lowest level octant (tif file)

        Arguments:
            start {3 array of ints} -- lower corner of rectangle
            end {3 array of ints} -- upper corner of rectangle (to be included)
            tree_specifics {list} -- path {string}, sz {3 array}, nl {int}, channel {0 or 1}
		
		Returns:
			volume {3-D array} -- image subvolume
        """
        start_path, start_coord = self.voxel_path(start, channel)
        end_path, end_coord = self.voxel_path(end, channel)
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

    def stitch_images(self, segments, channel):
        """Stitch together lowest level octants (tif files)

        Arguments:
            segments {3 list} -- segment locations as outputted by divide_voxel_bounds()

        Returns:
            volume {3-D array} -- image subvolume
        """
        if len(segments[0]) > 1:
            segments1 = segments[:]
            segments1[0] = [segments1[0][0]]

            segments2 = segments[:]
            segments2[0] = segments2[0][1:]

            left = self.stitch_images(segments1, channel)
            right = self.stitch_images(segments2, channel)

            return np.concatenate((left, right), axis=0)
        elif len(segments[1]) > 1:
            segments1 = segments[:]
            segments1[1] = [segments1[1][0]]

            segments2 = segments[:]
            segments2[1] = segments2[1][1:]

            left = self.stitch_images(segments1, channel)
            right = self.stitch_images(segments2, channel)

            return np.concatenate((left, right), axis=1)
        elif len(segments[2]) > 2:
            segments1 = segments[:]
            segments1[2] = [segments1[2][0]]

            segments2 = segments[:]
            segments2[2] = segments2[2][1:]

            left = self.stitch_images(segments1, channel)
            right = self.stitch_images(segments2, channel)

            return np.concatenate((left, right), axis=2)
        elif len(segments[2]) == 2:
            start1 = [segments[0][0][0], segments[1][0][0], segments[2][0][0]]
            end1 = [segments[0][0][1], segments[1][0][1], segments[2][0][1]]
            left = self.get_intrarectangle_voxel(start1, end1, channel)

            start2 = [segments[0][0][0], segments[1][0][0], segments[2][1][0]]
            end2 = [segments[0][0][1], segments[1][0][1], segments[2][1][1]]
            right = self.get_intrarectangle_voxel(start2, end2, channel)

            return np.concatenate((left, right), axis=2)
        else:
            start1 = [segments[0][0][0], segments[1][0][0], segments[2][0][0]]
            end1 = [segments[0][0][1], segments[1][0][1], segments[2][0][1]]
            left = self.get_intrarectangle_voxel(start1, end1, channel)
            return left

    def get_interrectangle_voxel(self, start, end, channel):
        """retrieve a 3d rectangle of image data given the voxel coordinates of the two corners

        Arguments:
            start {3-array of ints} -- start corner of rectangle
            end {3-array of ints} -- end corner of rectangle

        Returns:
            volume {3-D array} -- image subvolume
        """
        if any(end >= self.size):
            raise ValueError("Voxels requested are out of bounds")
        segments = self.divide_voxel_bounds(start, end)
        volume = self.stitch_images(segments, channel)
        return volume

    def get_rectangle_spatial(self, coord, radii, channel):
        """Finds a 3D rectangle of image data around a specified coordinate

            Arguments:
                coord {3-array} -- (x,y,z) spatial coordinates in um
                radii {3-array of ints} -- number of voxels around the coordinate to include, in the order x,y,z
                channel {0 or 1} -- desired image channel

            Returns:
                rectangle {3-D array} -- image subvolume

        """

        voxel_coord = self.space_to_voxel(coord)

        start_coord = voxel_coord - radii
        end_coord = voxel_coord + radii
        # modify ending coordinate
        if any(end_coord >= self.size):
            end_coord[end_coord >= self.size] = self.size[end_coord >= self.size]
        rectangle = self.get_interrectangle_voxel(start_coord, end_coord, channel)

        return rectangle

    def path_to_leftcoord(self, path):
        """Given a path to a tif file, output the voxel coordinate of its first entry

        Arguments:
            path {string or Path object} -- tif file path

        Returns:
            left {3-array} -- voxel coordinate of left corner (ie smallest coordinate) in a given tif image
        """
        sz_cur = self.size
        left = np.array([0, 0, 0])

        if isinstance(path, str):
            path = Path(path)
        string = str(path)
        dirs = string.split("/")
        dirs = dirs[-self.nl : -1]

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
