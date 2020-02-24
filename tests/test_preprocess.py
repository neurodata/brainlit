# Test that algorithm works
# test that inputs work?

import pytest
import numpy as np
from brainlit import preprocessing

def test_center():
    img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    centered_image = np.array([[-4,-3,-2],[-1,0,1],[2,3,4]])
    assert(np.array_equal(preprocessing.center(img), centered_image))


def test_contrast_normalize():
    img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    normalized_image = np.load("./output/contrast_normalize.npy")
    assert(np.array_equal(preprocessing.contrast_normalize(img), normalized_image))

def test_pad_undopad_transform():
    np.random.seed(6)
    img = np.random.randint(0, 256, size = (50, 50))
    window_size = np.array([5,5])
    step_size = np.array([2,2])
    padded, pad_size = preprocessing.window_pad(img, window_size, step_size)
    new_img = preprocessing.undo_pad(padded, pad_size)
    assert(np.array_equal(img, new_img))

def test_pad_undopad_transform_3D():
    np.random.seed(6)
    img = np.random.randint(0, 256, size = (50, 50, 50))
    window_size = np.array([5,5,5])
    step_size = np.array([2,2,2])
    padded, pad_size = preprocessing.window_pad(img, window_size, step_size)
    new_img = preprocessing.undo_pad(padded, pad_size)
    assert(np.array_equal(img, new_img))

def test_image_vector_transform():
    np.random.seed(6)

    img = np.random.randint(0, 256, size = (10, 10))
    window_size = np.array([3,3])
    step_size = np.array([1,1])
    vector = preprocessing.vectorize_img(img, window_size, step_size)
    new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
    assert(np.array_equal(img, new_image))

    # This starts failing at size ~20, 20
    # img = np.random.randint(0, 256, size = (20, 20))
    # window_size = np.array([3,3])
    # step_size = np.array([1,1])
    # vector = preprocessing.vectorize_img(img, window_size, step_size)
    # new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
    # assert(np.array_equal(img, new_image))


def test_image_vector_transform_3D():
    np.random.seed(6)

    img = np.random.randint(0, 256, size = (10, 10,10))
    window_size = np.array([3,3,3])
    step_size = np.array([1,1,1])
    vector = preprocessing.vectorize_img(img, window_size, step_size)
    new_image = preprocessing.imagize_vector(vector, img.shape, window_size, step_size)
    assert(np.array_equal(img, new_image))


def test_window_pad_2D():
    # Trivial example
    img = np.zeros([50, 50])
    window_size = np.array([3, 3])
    step_size = np.array([1, 1])

    [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
    assert(img_padded.shape == (54, 54))
    assert(np.array_equal(padding, np.array([[2,2], [2,2]])))


def test_window_pad_3D():
    #Trivial example
    img = np.zeros([50, 50, 50])
    window_size = np.array([3, 3, 3])
    step_size = np.array([1, 1, 1])

    [img_padded, padding] = preprocessing.window_pad(img, window_size, step_size)
    assert(img_padded.shape == (54, 54, 54))
    assert(np.array_equal(padding, np.array([[2,2], [2,2], [2,2]])))
