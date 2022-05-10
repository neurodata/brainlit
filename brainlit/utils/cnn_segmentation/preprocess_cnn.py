# preprocessing data from tifs to tensors for evaluation

from skimage import io
import numpy as np
from pathlib import Path
import os
from tqdm.notebook import tqdm
import torch
from torch.utils.data import DataLoader


def get_img_and_mask(data_dir):
    """Get lists of tif images and associated ground truth masks

    Arguments:
        data_dir: str, path to tif and mask files

    Returns:
        List of 3d np array images: X_img
        List of 3d np array masks: y_img
    """
    im_dir = Path(os.path.join(data_dir, "sample-tif-location"))
    gfp_files = list(im_dir.glob("**/*-gfp.tif"))
    X_img = []
    y_mask = []

    for i, im_path in enumerate(tqdm(gfp_files)):

        f = im_path.parts[-1][:-8].split("_")
        image = f[0]
        num = int(f[1])

        if (image == "test" and num in [9, 10, 24]) or (
            image == "validation" and num in [11]
        ):
            continue

        # getting image
        im = io.imread(im_path, plugin="tifffile")
        im = (im - np.amin(im)) / (np.amax(im) - np.amin(im))
        im = np.swapaxes(im, 0, 2)
        im_padded = np.pad(im, ((4, 4), (4, 4), (3, 3)))

        # getting ground truth mask
        file_name = (
            str(im_path)[
                str(im_path).find("\\", 80) + 1 : (str(im_path).find("sample"))
            ]
            + "/mask-location/"
        )
        file_num = file_name[file_name.find("_") + 1 :]
        if file_name[0] == "v":
            file_num = str(int(file_num) + 25)
        mask_path = Path(file_name + f[0] + "_" + f[1] + "_mask.npy")
        mask = np.load(mask_path)

        X_img.append(im)
        y_mask.append(mask)

    return X_img, y_mask


def train_test_split(X_img, y_mask, test_percent=0.25):
    """Get train/test/split of images and masks
    Args:
        X_img: list of 3d np array images
        y_mask: list of 3d np array masks

    Returns:
        Lists of specifie training and testing size: X_train, y_train, X_test, y_test: l
    """
    num_images = len(X_img)
    test_images = num_images * test_percent
    train_images = int(num_images - test_images)

    X_train = X_img[0:train_images]
    y_train = y_mask[0:train_images]

    X_test = X_img[train_images:num_images]
    y_test = y_mask[train_images:num_images]

    return X_train, y_train, X_test, y_test


def get_subvolumes(X_train, y_train, x_dim, y_dim, z_dim):
    """Get subvolumes of specified site for training dataset

    Arguments:
        X_train: list of imgs, from train_test_split function
        y_train: list of masks, from train_test_split function
        x_dim: int, x_dim of subvolume, must be divisible by image shape
        y_dim: int, y_dim of subvolume, must be divisible by image shape
        z_dim: int, z_dim of subvolume, must be divisible by image shape

    Returns:
        X_train_subvolume: List of image subvolumes, for training
        y_train_subvolume: List of associated mask subvolumes, for training
    """
    X_train_subvolumes = []
    y_train_subvolumes = []

    # getting subvolumes
    for image in X_train:
        i = 0
        while i < image.shape[0]:
            j = 0
            while j < image.shape[1]:
                k = 0
                while k < image.shape[2]:
                    subvol = image[i : i + x_dim, j : j + y_dim, k : k + z_dim]
                    X_train_subvolumes.append(subvol)
                    k += z_dim
                j += y_dim
            i += x_dim

    for mask in y_train:
        i = 0
        while i < mask.shape[0]:
            j = 0
            while j < mask.shape[1]:
                k = 0
                while k < mask.shape[2]:
                    subvol = mask[i : i + x_dim, j : j + y_dim, k : k + z_dim]
                    y_train_subvolumes.append(subvol)
                    k += z_dim
                j += y_dim
            i += x_dim

    return X_train_subvolumes, y_train_subvolumes


def getting_torch_objects(X_train_subvolumes, y_train_subvolumes, X_test, y_test):
    """Get training data in torch object format

    Arguments:
        X_train_subvolumes: list, training images (or subvolumes) from get_subvolumes function
        y_train_subvolumes: list, trianing masks (or subvolumes) from get_subvolumes function
        X_test: list, testing images from train_test_split function
        y_test: list, testing masks from train_test_split function

    Returns:
        List of image subvolumes for training: X_train_subvolume
        List of associated mask subvolumes for training: y_train_subvolumes
    """
    x_dim = X_train_subvolumes[0].shape[0]
    y_dim = X_train_subvolumes[0].shape[1]
    z_dim = X_train_subvolumes[0].shape[2]
    length = len(X_train_subvolumes)

    img_x_dim = X_test[0].shape[0]
    img_y_dim = X_test[0].shape[1]
    img_z_dim = X_test[0].shape[2]

    X_torch_train = np.reshape(X_train_subvolumes, (1, length, x_dim, y_dim, z_dim))
    y_torch_train = np.reshape(y_train_subvolumes, (1, length, x_dim, y_dim, z_dim))

    X_torch_test = np.reshape(X_test, (1, len(X_test), img_x_dim, img_y_dim, img_z_dim))
    y_torch_test = np.reshape(y_test, (1, len(y_test), img_x_dim, img_y_dim, img_z_dim))

    training_data = torch.tensor([X_torch_train, y_torch_train]).float()
    test_data = torch.tensor([X_torch_test, y_torch_test]).float()

    batch_size = 2
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # printing dataloader dimensions
    train_features, train_labels = next(iter(train_dataloader))
    print(f"Training features shape: {train_features.size()}")
    test_features, test_labels = next(iter(test_dataloader))
    print(f"Testing features shape: {test_features.size()}")

    # printing device torch is using (cuda or cpu)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    return train_dataloader, test_dataloader
