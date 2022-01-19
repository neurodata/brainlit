import pytest
import numpy as np
from brainlit.algorithms.connect_fragments import trace_evaluation
from numpy.testing import (
    assert_array_equal,
)

############################
### functionality checks ###
############################


def test_resample():
    a = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [4, 0, 0], [10, 0, 0]])

    func_resample = trace_evaluation.resample(a, spacing=1)
    true_resample = np.zeros([11, 3])
    true_resample[:, 0] = np.arange(0, 11)

    assert_array_equal(func_resample, true_resample)

    func_resample = trace_evaluation.resample(a, spacing=2)
    true_resample = np.zeros([7, 3])
    true_resample[:, 0] = [0, 1, 2, 4, 6, 8, 10]

    assert_array_equal(func_resample, true_resample)


def test_sd():
    a = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    b = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 2]])
    c = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 1]])
    d = np.array([[1, 1, 1], [0, 0, 0], [2, 2, 10]])

    sd = trace_evaluation.sd(a, b, substantial=False)
    assert sd == 0

    ssd = trace_evaluation.sd(a, b, substantial=True)
    assert ssd == 0

    sd = trace_evaluation.sd(a, c, substantial=False)
    assert sd == pytest.approx(1 / 3)

    ssd = trace_evaluation.sd(a, c, substantial=True)
    assert ssd == 0

    sd = trace_evaluation.sd(a, d, substantial=False)
    assert sd == pytest.approx(np.mean([8 / 3, np.sqrt(3) / 3]))

    ssd = trace_evaluation.sd(a, d, substantial=True)
    assert ssd == 8 / 2
