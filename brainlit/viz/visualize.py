import numpy as np
import matplotlib.pyplot as plt


def plot_image_2d(imgs, titles=None, rows=1, colorbar=False):
    """
        Plots a 2D image

        Parameters
        ----------
        imgs : single 2D np.array or list of 2D np.arrays
            slice of image data frog ngl_pipeline class.
            example use: img_slice = img[:,100,:]

        titles : str, default = None

        Returns
        -------
        matplotlib figure and axis
     """
    if isinstance(imgs, list):
        l = len(imgs)
        cols = np.ceil(l / rows).astype(int)
        fig, axes = plt.subplots(rows, cols)

        for i, img in enumerate(imgs):
            img = np.swapaxes(img, 0, 1)
            ax = axes[i]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if titles is not None:
                ax.set_title(titles[i])
        return fig, axes
    else:
        img = np.swapaxes(imgs, 0, 1)
        fig, axis = plt.subplots()
        im = axis.imshow(img)
        axis.set_xticks([])
        axis.set_yticks([])
        if colorbar:
            plt.colorbar(im, ax=axis)
        return fig, axis


def plot_image_mip(imgs, titles=None, rows=1, axis=2, colorbar=False):
    """
        Max Intensity Projection of 3D image

        Parameters
        ----------
        imgs : single 3D np.array or list of 3D np.arrays

        titles : str, default = None

        Returns
        -------
        matplotlib figure and axis
    """
    if isinstance(imgs, list):
        for i, img in enumerate(imgs):
            imgs[i] = np.amax(img, axis)
        fig, axes = plot_image_2d(imgs, rows=rows, titles=titles, colorbar=colorbar)
        return fig, axes
    else:
        mip = np.amax(imgs, axis)
        fig, axes = plot_image_2d(mip, titles=titles, colorbar=colorbar)
        return fig, axes


def find_smalldim(imgs):
    """
        Find smallest dimension of an image or list of images.

        Parameters
        ----------
        imgs : single 3D np.array or list of 3D np.arrays {

        titles : str, default = None

        Returns
        -------
        smallest_axis: int
    """
    if isinstance(imgs, list):
        img = imgs[0]
    else:
        img = imgs
    shp = img.shape
    smallest_axis = np.argmin(shp)
    return int(smallest_axis)


def plot_image_hist(imgs, rows=1, titles=None):
    """
        Histogram

        Parameters
        ----------
        imgs : single 3D np.array or list of 3D np.arrays {

        titles : str, default = None

        Returns
        -------
        matplotlib figure and axis
    """
    if isinstance(imgs, list):
        l = len(imgs)
        cols = np.ceil(l / rows).astype(int)
        fig, axes = plt.subplots(rows, cols)

        for i, img in enumerate(imgs):
            img = np.swapaxes(img, 0, 1)
            ax = axes[i]
            ax.hist(img.flatten(), bins=50)
            ax.set_ylabel("Count", fontsize=12)
            ax.set_xlabel("Intensity", fontsize=12)
            if titles is not None:
                ax.title(titles[i], fontsize=12)
        return fig, axes
    else:
        histo = plt.hist(imgs.flatten())
        plt.ylabel("Count", fontsize=12)
        plt.xlabel("Intensity", fontsize=12)
        return histo


def plot_image_pts(img, voxels, colorbar=False):
    """
        Visualize a chunk around a voxel

        Parameters
        ----------
        img : single 2D np.array

        voxels : np.array
            voxel output from ngl_pipeline class

        Returns
        -------
        matplotlib figure and axis
    """
    fig, axes = plt.subplots()

    if len(voxels.shape) > 1:
        im = axes.imshow(img)
        axes.scatter(voxels[:, 0], voxels[:, 1])
    else:
        im = axes.imshow(img)
        axes.scatter(voxels[0], voxels[1])

    if colorbar:
        plt.colorbar(im, ax=axes)

    return fig, axes
