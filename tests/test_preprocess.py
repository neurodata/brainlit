# Test that algorithm works
# test that inputs work?

import pytest
import numpy as np
import brainlit

def test_window_pad_2D():
    # Trivial example
    img = np.zeros([50, 50])
    window_size = np.array([3, 3])
    step_size = np.array([1, 1])

    [img_padded, padding] = brainlit.preprocessing.window_pad(img, window_size, step_size)
    assert(img_padded.shape == (54, 54))
    assert(np.array_equal(padding, np.array([[2,2], [2,2]])))


def test_window_pad_3D():
    #Trivial example
    img = np.zeros([50, 50, 50])
    window_size = np.array([3, 3, 3])
    step_size = np.array([1, 1, 1])

    [img_padded, padding] = brainlit.preprocessing.window_pad(img, window_size, step_size)
    assert(img_padded.shape == (54, 54, 54))
    assert(np.array_equal(padding, np.array([[2,2], [2,2], [2,2]])))
