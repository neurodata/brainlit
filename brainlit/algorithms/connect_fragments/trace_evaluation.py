import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min


def resample(path, spacing=1):
    """Resample a path (linearly) according to maximum distance between points

    Args:
        path (np.array): points of path, shaped as number points x number of features.
        spacing (int, optional): maximum distance between points. Defaults to 1.

    Returns:
        np.array: points of resampled path, shaped as number points x number of features.
    """
    new_path = []
    for n in np.arange(path.shape[0]):
        pt1 = path[n - 1 : n, :]
        pt2 = path[n : n + 1, :]

        new_path.append(pt1)
        dist = np.linalg.norm(pt1 - pt2)

        if dist > spacing:
            ts = np.arange(0, dist, spacing)
            mid = np.zeros((len(ts) - 1, 3))
            for i, t in enumerate(ts[1:]):
                mid[i, :] = pt1 + (t / dist) * (pt2 - pt1)
            new_path.append(mid)
    new_path.append(pt2)
    new_path = np.concatenate(new_path)
    return new_path


def sd(pts1, pts2, substantial=False):
    """Compute spatial distance between two paths according to Peng et. al. 2010.

    Args:
        pts1 (np.array): points of first path, shaped as number points x number of features.
        pts2 (np.array): points of second path, shaped as number points x number of features.
        substantial (bool, optional): whether to compute substantial spatial distance which ignores all points that have a closest point within 2 voxels. Defaults to False.
        verbose (bool, optional): [description]. Defaults to False.

    Returns:
       float : spatial distance or substantial spatial distance
    """
    _, dists1 = pairwise_distances_argmin_min(pts1, pts2)
    _, dists2 = pairwise_distances_argmin_min(pts2, pts1)
    if substantial:
        if any(dists1 > 2):
            ddiv1 = np.mean(dists1[dists1 > 2])
        else:
            ddiv1 = 0
        if any(dists2 > 2):
            ddiv2 = np.mean(dists2[dists2 > 2])
        else:
            ddiv2 = 0

        return np.mean([ddiv1, ddiv2])
    else:
        ddiv1 = np.mean(dists1)
        ddiv2 = np.mean(dists2)
        return np.mean([ddiv1, ddiv2])
