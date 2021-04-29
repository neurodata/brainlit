import numpy as np
from brainlit.algorithms.detect_somas import find_somas
from pytest import raises

##############
### inputs ###
##############


def test_find_somas_bad_input():
    volume = []
    res = ""

    # test input volume is numpy.ndarray
    with raises(TypeError, match=r"should be <class 'numpy.ndarray'>"):
        find_somas(volume, res)
    volume = np.array([1, "a", 2])
    with raises(
        TypeError, match=r"elements should be \(<class 'int'>, <class 'float'>\)."
    ):
        find_somas(volume, res)
    volume = np.zeros((2, 2))
    # test input volume must be three-dimensional
    with raises(ValueError, match=r"Input volume must be three-dimensional"):
        find_somas(volume, res)
    volume = np.zeros((19, 19, 21))
    # test input volume has to be at least 20x20xNz
    with raises(ValueError, match=r"Input volume is too small"):
        find_somas(volume, res)
    volume = np.zeros((50, 50, 50))

    # test res should be list
    with raises(TypeError, match=r"should be <class 'list'>"):
        find_somas(volume, res)
    res = [1, "a", 2]
    with raises(
        TypeError, match=r"elements should be \(<class 'int'>, <class 'float'>\)."
    ):
        find_somas(volume, res)
    res = [100, 100.0]
    with raises(ValueError, match=r"Resolution must be three-dimensional"):
        find_somas(volume, res)
    res = [100, 100.0, 0]
    with raises(ValueError, match=r"Resolution must be non-zero at every position"):
        find_somas(volume, res)
