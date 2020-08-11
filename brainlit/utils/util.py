import contextlib
import joblib
from pathlib import Path
from typing import Optional, List, Tuple
from tqdm import tqdm
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from skimage import draw


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report into tqdm progress bar given as argument
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def check_type(input, types):
    if not isinstance(input, types):
        raise TypeError((f"{input} should be {types}, not {type(input)}."))


def check_iterable_type(input, types):
    if not all(isinstance(i, types) for i in input):
        raise TypeError((f"{input} elements should be {types}."))


def check_iterable_positive(input):
    if not all(i > 0 for i in input):
        raise ValueError((f"{input} elements should be positive."))


def check_iterable_nonnegative(input):
    if not all(i >= 0 for i in input):
        raise ValueError((f"{input} elements should be nonnegative."))


def check_size(input, allow_float=True):
    check_type(input, (list, tuple))
    if len(input) != 3:
        raise ValueError(f"{input} must have x, y, z dimensions")
    if allow_float:
        check_iterable_type(input, (int, float))
    else:
        check_iterable_type(input, int)


def check_precomputed(input):
    check_type(input, str)
    prefix = input.split(":")[0]
    if prefix not in ["file", "s3", "gc"]:
        raise NotImplementedError("only file, s3, and gc prefixes supported")


def check_binary_path(input):
    check_iterable_type(input, str)  # ensure list of str
    for bcode in input:
        if not all(c in "01" for c in bcode):
            raise ValueError(f"Binary paths are made of 0s and 1s, not like {bin}")


def tubes_from_paths(
    size: Tuple[int, int, int], paths: List[List[int]], radius: Optional[int] = None
):
    """Constructs tubes from list of paths.
    Returns densely labeled paths within the shape of the image.

    Arguments:
        size: The size of image to consider.
        paths: The list of paths. Each path is a list of points along the path (non-dense).
        radius: The radius of the line to draw. Default is None = 1 pixel wide line.
    """

    def _within_img(line, size):
        arrline = np.array(line).astype(int)
        arrline = arrline[:, arrline[0, :] < size[0]]
        arrline = arrline[:, arrline[0, :] >= 0]
        arrline = arrline[:, arrline[1, :] < size[1]]
        arrline = arrline[:, arrline[1, :] >= 0]
        arrline = arrline[:, arrline[2, :] < size[2]]
        arrline = arrline[:, arrline[2, :] >= 0]
        return (arrline[0, :], arrline[1, :], arrline[2, :])

    coords = [[], [], []]
    for path in tqdm(paths):
        for i in range(len(path) - 1):
            line = draw.line_nd(path[i], path[i + 1])
            line = _within_img(line, size)
            if len(line) > 0:
                coords[0] = np.concatenate((coords[0], line[0]))
                coords[1] = np.concatenate((coords[1], line[1]))
                coords[2] = np.concatenate((coords[2], line[2]))

    try:
        coords = (coords[0].astype(int), coords[1].astype(int), coords[2].astype(int))
    except AttributeError:
        coords = (coords[0], coords[1], coords[2])

    if radius is not None:
        line_array = np.ones(size, dtype=int)
        line_array[coords] = 0
        seg = distance_transform_edt(line_array)
        labels = np.where(seg <= radius, 1, 0)
    else:
        labels = np.zeros(size, dtype=int)
        labels[coords] = 1

    return labels
