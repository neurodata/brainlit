import pytest
import numpy as np
from brainlit.preprocessing.image_process import (
    gabor_filter,
    getLargestCC,
    removeSmallCCs,
    label_points,
    compute_frags,
)
from brainlit.preprocessing.preprocess import (
    center,
    contrast_normalize,
    whiten,
    window_pad,
    undo_pad,
    vectorize_img,
    imagize_vector,
)
from numpy.testing import (
    assert_equal,
    assert_allclose,
    assert_array_equal,
    assert_almost_equal,
    assert_array_almost_equal,
)


####################
### input checks ###
####################


def test_whiten_bad_inputs():
    # test window_size.ndim > 1 or step_size.ndim > 1
    with pytest.raises(ValueError):
        whiten(
            img=np.ones((5, 5)),
            window_size=np.ones((5, 5)),
            step_size=np.array([3, 3]),
            centered=False,
            epsilon=1e-5,
            type="PCA",
        )
    with pytest.raises(ValueError):
        whiten(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3]),
            step_size=np.ones((5, 5)),
            centered=False,
            epsilon=1e-5,
            type="PCA",
        )

    # test len(window_size) != len(step_size)
    with pytest.raises(ValueError):
        whiten(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3]),
            step_size=np.array([3, 3, 3]),
            centered=False,
            epsilon=1e-5,
            type="PCA",
        )

    # test img.ndim != len(window_size)
    with pytest.raises(ValueError):
        whiten(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3, 3]),
            step_size=np.array([3, 3, 3]),
            centered=False,
            epsilon=1e-5,
            type="PCA",
        )

    # test 'type' must be string ('PCA' or 'ZCA')
    with pytest.raises(ValueError):
        whiten(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3]),
            step_size=np.array([3, 3]),
            centered=False,
            epsilon=1e-5,
            type="nonsensical",
        )


def test_window_pad_bad_inputs():
    # test window_size.ndim > 1 or step_size.ndim > 1
    with pytest.raises(ValueError):
        window_pad(
            img=np.ones((5, 5)), window_size=np.ones((5, 5)), step_size=np.array([3, 3])
        )
    with pytest.raises(ValueError):
        window_pad(
            img=np.ones((5, 5)), window_size=np.array([3, 3]), step_size=np.ones((5, 5))
        )

    # test len(window_size) != len(step_size)
    with pytest.raises(ValueError):
        window_pad(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3]),
            step_size=np.array([3, 3, 3]),
        )

    # test img.ndim != len(window_size)
    with pytest.raises(ValueError):
        window_pad(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3, 3]),
            step_size=np.array([3, 3, 3]),
        )


def test_undo_pad_bad_inputs():
    # test pad_size.ndim == 1 and img.ndim != 1
    with pytest.raises(ValueError):
        undo_pad(img=np.ones((5, 5)), pad_size=np.ones((5)))

    # test img.ndim != pad_size.shape[0]
    with pytest.raises(ValueError):
        undo_pad(img=np.ones((5, 5)), pad_size=np.ones((5, 5)))


def test_vectorize_img_bad_inputs():
    # test window_size.ndim > 1 or step_size.ndim > 1
    with pytest.raises(ValueError):
        vectorize_img(
            img=np.ones((5, 5)), window_size=np.ones((5, 5)), step_size=np.array([3, 3])
        )
    with pytest.raises(ValueError):
        vectorize_img(
            img=np.ones((5, 5)), window_size=np.array([3, 3]), step_size=np.ones((5, 5))
        )

    # test len(window_size) != len(step_size)
    with pytest.raises(ValueError):
        vectorize_img(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3]),
            step_size=np.array([3, 3, 3]),
        )

    # test img.ndim != len(window_size)
    with pytest.raises(ValueError):
        vectorize_img(
            img=np.ones((5, 5)),
            window_size=np.array([3, 3, 3]),
            step_size=np.array([3, 3]),
        )


def test_imagize_vector_bad_inputs():
    # test window_size.ndim > 1 or step_size.ndim > 1
    with pytest.raises(ValueError):
        imagize_vector(
            img=np.ones((5, 5)),
            orig_shape=np.array([3, 3]),
            window_size=np.ones((5, 5)),
            step_size=np.array([3, 3]),
        )
    with pytest.raises(ValueError):
        imagize_vector(
            img=np.ones((5, 5)),
            orig_shape=np.array([3, 3]),
            window_size=np.array([3, 3]),
            step_size=np.ones((5, 5)),
        )

    # test len(window_size) != len(step_size)
    with pytest.raises(ValueError):
        imagize_vector(
            img=np.ones((5, 5)),
            orig_shape=np.array([3, 3]),
            window_size=np.array([3, 3]),
            step_size=np.array([3, 3, 3]),
        )

    # test len(orig_shape) != len(window_size)
    with pytest.raises(ValueError):
        imagize_vector(
            img=np.ones((5, 5)),
            orig_shape=np.array([3, 3, 3]),
            window_size=np.array([3, 3]),
            step_size=np.array([3, 3]),
        )


def test_gabor_filter_bad_inputs():
    """Tests that errors are raised when bad inputs are given to image_process.gabor_filter."""
    with pytest.raises(TypeError):
        gabor_filter(input=0, sigma=0, phi=0, frequency=5)
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma="a", phi=0, frequency=5)
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi="a", frequency=5)
    with pytest.raises(RuntimeError):
        gabor_filter(input=np.ones((5, 5)), sigma=[1, 1, 1], phi=0, frequency=5)
    with pytest.raises(RuntimeError):
        gabor_filter(input=np.ones((5, 5)), sigma=[1, 1], phi=[1, 1], frequency=5)
    with pytest.raises(RuntimeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output="a")
    with pytest.raises(RuntimeError):
        gabor_filter(
            input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output=np.ones((3, 3))
        )
    with pytest.raises(RuntimeError):
        gabor_filter(
            input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, mode="nonsensical"
        )
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, cval="a")
    with pytest.raises(TypeError):
        gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, truncate="a")


def test_gabor_filter_valid_inputs():
    """Tests that no errors are raised when valid inputs are given to image_process.gabor_filter."""
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1.0, phi=1, frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=np.dtype("float32").type(1), phi=1, frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=[1, 1], phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=[1.0, 1.0], phi=1, frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=np.ones(2).astype("float32"), phi=1, frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1.0, frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=np.dtype("float32").type(1), frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=[1], frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=[1.0], frequency=1)
    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=np.ones(1).astype("float32"), frequency=1
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1.0)
    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=1, frequency=np.dtype("float32").type(1)
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, offset=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, offset=1.0)
    gabor_filter(
        input=np.ones((5, 5)),
        sigma=1,
        phi=1,
        frequency=1,
        offset=np.dtype("float32").type(1),
    )

    gabor_filter(
        input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output=np.ones((5, 5))
    )
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, output="float32")

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, cval=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, cval=1.0)
    gabor_filter(
        input=np.ones((5, 5)),
        sigma=1,
        phi=1,
        frequency=1,
        cval=np.dtype("float32").type(1),
    )

    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, truncate=1)
    gabor_filter(input=np.ones((5, 5)), sigma=1, phi=1, frequency=1, truncate=1.0)
    gabor_filter(
        input=np.ones((5, 5)),
        sigma=1,
        phi=1,
        frequency=1,
        truncate=np.dtype("float32").type(1),
    )


def test_getLargestCC_bad_input():
    with pytest.raises(TypeError):
        getLargestCC(1)
    with pytest.raises(ValueError):
        getLargestCC(np.zeros((5, 5)))


def test_getLargestCC_valid_input():
    getLargestCC(np.ones((5, 5)))


def test_removeSmallCCs_bad_input():
    with pytest.raises(TypeError):
        removeSmallCCs(1, 2)
    with pytest.raises(TypeError):
        removeSmallCCs(np.ones((5, 5)), [2])


def test_removeSmallCCs_valid_input():
    removeSmallCCs(np.ones((5, 5)), 2)


def test_label_points_valid_input():
    labels = np.zeros((10, 10, 10), dtype=int)
    labels[0, 0, 0] = 1
    labels[9, 9, 9] = 2
    points = [[1, 1, 1], [8, 8, 8]]
    res = [1, 1, 1]

    label_points(labels, points, res)


def test_compute_frags_valid_input():
    soma_coords = [[0, 0, 0]]
    labels = np.zeros((10, 10, 10), dtype=int)
    labels[0, 0, 0] = 1
    labels[9, 9, 9] = 2

    im_processed = 0.2 * np.ones((10, 10, 10))
    im_processed[0, 0, 0] = 0.91
    im_processed[9, 9, 9] = 0.95

    threshold = 0.9

    res = [1, 1, 1]

    compute_frags(soma_coords, labels, im_processed, threshold, res)
    compute_frags(
        soma_coords, labels, im_processed, threshold, res, chunk_size=[5, 5, 5], ncpu=2
    )


############################
### functionality checks ###
############################


def test_center():
    arr = np.array([1, 2, 6])
    centered = center(arr)
    answer = np.array([-2, -1, 3])
    assert_array_equal(centered, answer)


def test_contrast_normalize():
    arr = np.array([1, 5])
    normalized = contrast_normalize(arr)
    answer = np.array([-1, 1])
    assert_array_equal(normalized, answer)


def test_window_pad():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    window_size = np.array([3, 3])
    step_size = np.array([2, 2])
    padded = window_pad(img, window_size, step_size)
    answer = np.array(
        [
            [1, 1, 2, 3, 3],
            [1, 1, 2, 3, 3],
            [4, 4, 5, 6, 6],
            [7, 7, 8, 9, 9],
            [7, 7, 8, 9, 9],
        ]
    )
    pad_size = np.array([[1, 1], [1, 1]])  # [[top,bottom], [left, right]]
    assert_equal(padded, (answer, pad_size))


def test_undo_pad():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    pad_size = np.array([[0, 1], [0, 1]])  # [[top, bottom], [left, right]]
    unpadded = undo_pad(img, pad_size)
    answer = np.array([[1, 2], [4, 5]])
    assert_array_equal(unpadded, answer)


def test_vectorize_img():
    img = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    window_size = np.array([2, 2])
    step_size = np.array([1, 1])
    vectorized = vectorize_img(img, window_size, step_size)
    answer = np.array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]])
    assert_array_equal(vectorized, answer)


def test_imagize_vector():
    """2d example using the same example as test_vectorize_img."""
    img = np.array([[1, 2, 4, 5], [2, 3, 5, 6], [4, 5, 7, 8], [5, 6, 8, 9]])
    orig_shape = (3, 3)
    window_size = np.array([2, 2])
    step_size = np.array([1, 1])
    imagized = imagize_vector(img, orig_shape, window_size, step_size)
    answer = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    assert_array_equal(imagized, answer)


def test_gaussian_truncate():
    """Tests that Gaussian filters can be truncated at different widths."""
    arr = np.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (np.abs(gabor_filter(arr, 5, 0, 5, truncate=2)[0]) > 0).sum()
    assert_equal(num_nonzeros_2, 21**2)
    num_nonzeros_5 = (np.abs(gabor_filter(arr, 5, 0, 5, truncate=5)[0]) > 0).sum()
    assert_equal(num_nonzeros_5, 51**2)

    f = np.abs(gabor_filter(arr, [0.5, 2.5], 0, 5, truncate=3.5)[0])
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    # n0 should be 2*int(2.5*3.5 + 0.5) + 1
    assert_equal(n0, 19)
    n1 = fpos.any(axis=1).sum()
    # n1 should be 2*int(0.5*3.5 + 0.5) + 1
    assert_equal(n1, 5)


def test_gabor_filter():
    """Tests that the gabor filter works correctly."""
    # Test generic case
    input = np.array([[1, 2, 3], [2, 4, 6]], np.float32)
    real, imag = gabor_filter(input, 1.0, 0, 5)
    assert_equal(input.dtype, real.dtype)
    assert_equal(input.shape, real.shape)
    assert_equal(input.dtype, imag.dtype)
    assert_equal(input.shape, imag.shape)

    # Test using otype
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    otype = np.float64
    real, imag = gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
    assert_equal(real.dtype.type, np.float64)
    assert_equal(input.shape, real.shape)
    assert_equal(imag.dtype.type, np.float64)
    assert_equal(input.shape, imag.shape)

    # Tests that inputting an output array works
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    output = np.zeros([100, 100]).astype(np.float32)
    gabor_filter(input, [1.0, 1.0], 0, 5, output=output)
    assert_equal(input.dtype, output.dtype)
    assert_equal(input.shape, output.shape)

    # Tests that list of sigmas is same as single
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    otype = np.float64
    real1, imag1 = gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
    real2, imag2 = gabor_filter(input, 1.0, 0, 5, output=otype)
    assert_array_almost_equal(real1, real2)
    assert_array_almost_equal(imag1, imag2)

    # Tests that list of phi works
    input = np.arange(100 * 100 * 100).astype(np.float32)
    input.shape = (100, 100, 100)
    otype = np.float64
    real1, imag1 = gabor_filter(input, 1.0, [0.5, 0.5], 5, output=otype)
    real2, imag2 = gabor_filter(input, 1.0, 0.5, 5, output=otype)
    assert_array_almost_equal(real1, real2)
    assert_array_almost_equal(imag1, imag2)


def test_getLargestCC():
    """Tests that getLargestCC works correctly."""
    img = np.zeros([50, 50])
    img[25, 25] = 1
    output = getLargestCC(img)
    assert_array_equal(img, output)

    img[25, 26] = 1
    img[0, 0] = 1
    expected_output = img
    expected_output[0, 0] = 0

    output = getLargestCC(img)
    assert_array_equal(expected_output, output)


def test_removeSmallCCs():
    """Tests that removeSmallCCs works correctly."""
    img = np.zeros([50, 50])
    img[25, 25] = 1
    img[25, 26] = 1
    output = removeSmallCCs(img, 1)
    assert_array_equal(img, output)

    img = np.zeros([50, 50])
    expected_output = np.zeros([50, 50])
    for i in range(0, 2):
        for j in range(0, 2):
            img[i, j] = 1
    for i in range(4, 7):
        for j in range(4, 7):
            img[i, j] = 1
            expected_output[i, j] = 1
    for i in range(20, 30):
        for j in range(20, 30):
            img[i, j] = 1
            expected_output[i, j] = 1
    output = removeSmallCCs(img, 5).astype(int)
    assert_array_equal(expected_output, output)


def test_label_points():
    labels = np.zeros((10, 10, 10), dtype=int)
    labels[0, 0, 0] = 1
    labels[9, 9, 9] = 2
    points = [[9, 9, 0], [0, 0, 9]]
    res = [0.01, 0.01, 1]

    points, point_labels = label_points(labels, points, res)
    expected_output = [1, 2]

    assert_array_equal(expected_output, point_labels)


def test_compute_frags():
    soma_coords = [[0, 0, 0]]
    labels = np.zeros((100, 100, 100), dtype=int)
    labels[0:5, 0:5, 0:10] = 1
    labels[10:15, 10:15, 20:40] = 2
    labels[50:55, 50:55, 30:80] = 3

    im_processed = 0.2 * np.ones((100, 100, 100))
    im_processed[0:5, 0:5, 0:10] = 0.91
    im_processed[10:15, 10:15, 20:40] = 0.93
    im_processed[50:55, 50:55, 30:80] = 0.94

    threshold = 0.9

    res = [1, 1, 1]

    new_labels = compute_frags(soma_coords, labels, im_processed, threshold, res)
    assert len(np.unique(new_labels)) > 8

    new_labels = compute_frags(
        soma_coords,
        labels,
        im_processed,
        threshold,
        res,
        chunk_size=[50, 50, 50],
        ncpu=2,
    )
    assert len(np.unique(new_labels)) > 8
