import numpy as np
import matplotlib.pyplot as plt


def plot_image_2d(imgs, titles=None, rows=1, colorbar=False):
    """Plot image using matplotlib
    
    Arguments:
        img {numpy array} -- image to be displayed

    Keyword Arguments:
        rows {int} -- number of rows in the subplot
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
    """max intensity projection
    
    Arguments:
        imgs {3d array or list of 3d arrays} -- image(s)
    
    Keyword Arguments:
        titles {list of strings} -- titles for the images (default: None)
        axis {int} -- axis along which to project (default: {2})
        rows {int} -- number of rows in the subplot (default: 1)
        colorbar {bool} -- whether a colorbar should be included (default: False)
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
    """Find the smallest dimension of the image

    Arguments:
        imgs {3d array or list of 3d arrays} -- image(s)
    
    """
    if isinstance(imgs, list):
        img = imgs[0]
    else:
        img = imgs
    shp = img.shape
    smallest_axis = np.argmin(shp)
    return int(smallest_axis)


def plot_image_hist(imgs, rows=1, titles=None):
    """Histograms
    
    Arguments:
        img {3d array} -- images from which to plot a intensity histogram
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
    """ Visualize a chunk around a voxel 
    Arguments:
        img {2D array} 
        voxels: 2D array 
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
