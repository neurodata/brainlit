import pytest
import numpy as np
from brainlit import preprocessing as pp 
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_equal, assert_almost_equal,
                           assert_array_almost_equal)


def test_gaussian_truncate():
    # Test that Gaussian filters can be truncated at different widths.
    # These tests only check that the result has the expected number
    # of nonzero elements.
    arr = np.zeros((100, 100), float)
    arr[50, 50] = 1
    num_nonzeros_2 = (np.abs(pp.gabor_filter(arr, 5, 0, 5, truncate = 2)) > 0).sum()
    assert_equal(num_nonzeros_2, 21**2)
    num_nonzeros_5 = (np.abs(pp.gabor_filter(arr, 5, 0, 5, truncate = 5)) > 0).sum()
    assert_equal(num_nonzeros_5, 51**2)

    f = np.abs(pp.gabor_filter(arr, [0.5, 2.5], 0, 5, truncate = 3.5))
    fpos = f > 0
    n0 = fpos.any(axis=0).sum()
    # n0 should be 2*int(2.5*3.5 + 0.5) + 1
    assert_equal(n0, 19)
    n1 = fpos.any(axis=1).sum()
    # n1 should be 2*int(0.5*3.5 + 0.5) + 1
    assert_equal(n1, 5)

def test_gabor_filter():
    # Test generic case
    input = np.array([[1, 2, 3],
                      [2, 4, 6]], np.float32)
    output = pp.gabor_filter(input, 1.0, 0, 5)
    assert_equal(input.dtype, output.dtype)
    assert_equal(input.shape, output.shape)

    # Test using otype
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    otype = np.float64
    output = pp.gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
    assert_equal(output.dtype.type, np.float64)
    assert_equal(input.shape, output.shape)

    # Tests that inputting an output array works
    input = np.arange(100 * 100).astype(np.float64)
    input.shape = (100, 100)
    pp.gabor_filter(input, [1.0, 1.0], 0, 5, output=output)
    assert_equal(input.dtype, output.dtype)
    assert_equal(input.shape, output.shape)

    # Tests that list of sigmas is same as single
    input = np.arange(100 * 100).astype(np.float32)
    input.shape = (100, 100)
    otype = np.float64
    output1 = pp.gabor_filter(input, [1.0, 1.0], 0, 5, output=otype)
    output2 = pp.gabor_filter(input, 1.0, 0, 5, output=otype)
    assert_array_almost_equal(output1, output2)

    # Tests that list of phi works
    input = np.arange(100 * 100 * 100).astype(np.float32)
    input.shape = (100, 100, 100)
    otype = np.float64
    output1 = pp.gabor_filter(input, 1.0, [0.5, 0.5], 5, output=otype)
    output2 = pp.gabor_filter(input, 1.0, 0.5, 5, output=otype)
    assert_array_almost_equal(output1, output2)
