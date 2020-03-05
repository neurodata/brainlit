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