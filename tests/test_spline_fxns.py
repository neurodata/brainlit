import numpy as np
from brainlit.algorithms.generate_fragments import spline_fxns
from pytest import approx


def test_splev_deg0():
    x = np.array([-0.5, 0, 0.5, 1])
    xi = np.array([-1, 0, 1, 2])
    b0 = spline_fxns.splev_deg0(x, xi, 1)
    assert (b0 == np.array([0, 1, 1, 0])).all()

    xi = np.array([-1, 0, 1])
    b0 = spline_fxns.splev_deg0(x, xi, 1)
    assert (b0 == np.array([0, 1, 1, 1])).all()

def test_splev_degreecontrol():
    x = np.linspace(-1, 1, 4)
    # TEST p < 0
    tck = tuple([[], [], -1])
    b = spline_fxns.splev_degreecontrol(x, tck)
    assert(b == np.zeros(len(x))).all()

    # TEST p = 0
    xi = np.array([0, 0.5, 1])
    cs = np.array([1, 1])
    p = 0
    tck = tuple([xi, cs, p])
    b = spline_fxns.splev_degreecontrol(x, tck)
    assert(b == np.array([0, 0, 1, 1])).all()

    # TEST p > 0
