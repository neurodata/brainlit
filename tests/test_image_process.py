import pytest
import numpy as np
from brainlit import preprocessing as pp 
from numpy.testing import (assert_equal, assert_allclose,
                           assert_array_equal, assert_almost_equal)

def test_multiple_modes():
    # Test that the filters with multiple mode cababilities for different
    # dimensions give the same result as applying a single mode.
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    mode1 = 'reflect'
    mode2 = ['reflect', 'reflect']

    assert_equal(pp.gabor_filter(arr, 1, 0, 5, mode=mode1),
                 pp.gabor_filter(arr, 1, 0, 5, mode=mode1))

def test_multiple_modes_gabor():
    # Test gabor filter for multiple extrapolation modes
    arr = np.array([[1., 0., 0.],
                    [1., 1., 0.],
                    [0., 0., 0.]])

    expected = np.array([[-0.28438687, 0.01559809, 0.19773499],
                         [-0.36630503, -0.20069774, 0.07483620],
                         [0.15849176, 0.18495566, 0.21934094]])

    modes = ['reflect', 'wrap']

    assert_almost_equal(expected,
                        sndi.gaussian_laplace(arr, 1, mode=modes))