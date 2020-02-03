import numpy as np


def center(data):
    """Centers data by subtracting the mean
    
    Arguments:
        data {numpy arary of any shape/size} -- data to be centered

    Returns:
        data_centered -- centered data
    """
    data_centered = data - np.mean(data)
    return data_centered


def contrast_normalize(data, centered=False):
    """Make variance of data one
    
    Arguments:
        data {numpy array of any shape/size} -- data to be normalized
    
    Keyword Arguments:
        centered {bool} -- whether the data has already been centered (default: {False})

    Returns:
        data -- normalized data
    """
    if not centered:
        data = center(data)
    data = np.divide(data, np.sqrt(np.var(data)))
    return data


def whiten(data, window_size, step_size, centered=False):
    if not centered:
        data = center(data)

    data_padded, pad_size = window_pad(data, window_size, step_size)
    data_vectorized = vectorize_img(data_padded, window_size, step_size)

    c = np.cov(data_vectorized)
    U, S, V = np.linalg.svd(c)
    eps = 1e-5

    whiten_matrix = np.dot(np.diag(1.0 / np.sqrt(S + eps)), U.T)
    whitened = np.dot(whiten_matrix, data_vectorized)

    data_whitened = imagize_vector(whitened, data_padded.shape, window_size, step_size)
    data_whitened = undo_pad(data_whitened, pad_size)

    return data_whitened, S


def undo_pad(data, pad_size):
    start = pad_size[:, 0].astype(int)
    end = (data.shape - pad_size[:, 1]).astype(int)
    coords = list(zip(start, end))
    slices = tuple(slice(coord[0], coord[1]) for coord in coords)
    data = data[slices]

    return data


def window_pad(img, window_size, step_size):
    """Pad image at edges so the window can convolve evenly. 
    Padding will be a copy of the edges.
    
    Arguments:
        img {array} -- image to be padded
        window_size {array} -- window size that will be convolved
        step_size {array} -- step size of the windows

    Returns:
        img_padded -- 
        pad_size --
    """
    shp = img.shape
    d = len(shp)

    pad_size = np.zeros([d, 2])
    pad_size[:, 0] = window_size - 1

    num_steps = np.floor(np.divide(shp + window_size - 2, step_size))

    final_loc = np.multiply(num_steps, step_size)

    pad_size[:, 1] = final_loc - shp + 1

    pad_width = [pad_size[dim, :].astype(int).tolist() for dim in range(d)]

    img_padded = np.pad(img, pad_width, mode="edge")

    return img_padded, pad_size


def vectorize_img(img, window_size, step_size):
    shp = img.shape

    num_steps = (np.floor(np.divide(shp - window_size, step_size)) + 1).astype(int)
    vectorized = np.zeros([np.product(window_size), np.product(num_steps)])

    for step_num, step_coord in enumerate(np.ndindex(*num_steps)):
        start = np.multiply(step_coord, step_size)
        end = start + window_size

        coords = list(zip(start, end))
        slices = tuple(slice(coord[0], coord[1]) for coord in coords)
        """
        print(num_steps)
        print(step_num)
        print(slices)
        """
        vectorized[:, step_num] = img[slices].flatten()

    return vectorized


def imagize_vector(data, orig_shape, window_size, step_size):
    imagized = np.zeros(orig_shape)
    d = len(orig_shape)

    shp = orig_shape

    num_steps = (np.floor(np.divide(shp - window_size, step_size)) + 1).astype(int)
    vectorized = np.zeros([np.product(window_size), np.product(num_steps)])

    for step_num, step_coord in enumerate(np.ndindex(*num_steps)):
        start = np.multiply(step_coord, step_size)
        end = start + window_size

        coords = list(zip(start, end))
        slices = tuple(slice(coord[0], coord[1]) for coord in coords)

        imagized_temp = np.zeros(orig_shape)
        imagized_temp = data[:, step_num].reshape(window_size)
        stacked = np.stack((imagized[slices], imagized_temp), axis=-1)
        imagized[slices] = np.true_divide(stacked.sum(d), (stacked != 0).sum(d))

    return imagized
