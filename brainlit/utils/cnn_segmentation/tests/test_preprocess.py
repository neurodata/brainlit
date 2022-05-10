import pytest

import numpy as np
from brainlit.utils.cnn_segmentation import preprocess
from numpy.testing import (
    assert_array_equal,
)

############################
### functionality checks ###
############################

def test_train_test_split():
    X_img = [0, 1, 2, 3]
    y_mask = [0.0, 1.1, 2.2, 3.3]


    X_train, y_train, X_test, y_test = preprocess.train_test_split(X_img, y_mask)

    X_train_true = [0, 1, 2]
    y_train_true = [0.0, 1.1, 2.2]
    X_test_true = [3]
    y_test_true = [3.3]

    assert_array_equal(X_train, X_train_true)
    assert_array_equal(y_train, y_train_true)
    assert_array_equal(X_test, X_test_true)
    assert_array_equal(y_test, y_test_true)


def test_get_subvolumes():
    X_train = [np.zeros(shape=(4, 4, 4))]
    y_train = [np.ones(shape=(4, 4, 4))]

    x_dim = 2
    y_dim = 2
    z_dim = 2

    X_train_subvolumes, y_train_subvolumes = preprocess.get_subvolumes(X_train, y_train, x_dim, y_dim, z_dim)

    X_train_subvolumes_true = [np.zeros(shape=(2, 2, 2)), np.zeros(shape=(2, 2, 2)), np.zeros(shape=(2, 2, 2)),
                               np.zeros(shape=(2, 2, 2)), np.zeros(shape=(2, 2, 2)), np.zeros(shape=(2, 2, 2)),
                               np.zeros(shape=(2, 2, 2)), np.zeros(shape=(2, 2, 2))]

    y_train_subvolumes_true = [np.ones(shape=(2, 2, 2)), np.ones(shape=(2, 2, 2)), np.ones(shape=(2, 2, 2)),
                               np.ones(shape=(2, 2, 2)), np.ones(shape=(2, 2, 2)), np.ones(shape=(2, 2, 2)),
                               np.ones(shape=(2, 2, 2)), np.ones(shape=(2, 2, 2))]

    assert_array_equal(X_train_subvolumes[0], X_train_subvolumes_true[0])
    assert_array_equal(y_train_subvolumes[0], y_train_subvolumes_true[0])
    
    
def test_getting_torch_objects():
    X_train = [np.zeros(shape=(4, 4, 4))]
    y_train = [np.ones(shape=(4, 4, 4))]
    X_test = [np.zeros(shape=(4, 4, 4))]
    y_test = [np.ones(shape=(4, 4, 4))]

    train_dataloader, test_dataloader = preprocess.getting_torch_objects(X_train, y_train, X_test, y_test)

    train_dataloader_size = [2, 1, 1, 4, 4, 4]
    test_dataloader_size = [2, 1, 1, 4, 4, 4]

    assert_array_equal(list(next(iter(train_dataloader)).size()), train_dataloader_size)
    assert_array_equal(list(next(iter(test_dataloader)).size()), test_dataloader_size)

    




    
