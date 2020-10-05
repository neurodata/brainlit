import numpy as np
from brainlit.algorithms.generate_fragments import spline_fxns
from pytest import approx, raises
from scipy.interpolate import splprep, splev


def test_splev_deg0():
    x = []
    xi = []
    i = 1.5
    with raises(
        TypeError, match=r".* should be \(<class 'int'>, <class 'numpy.integer'>\).*"
    ):
        spline_fxns.splev_deg0(x, xi, i)

    i = 1
    with raises(ValueError, match="xi cannot be empty"):
        spline_fxns.splev_deg0(x, xi, i)

    xi = ["a", "b", "c"]
    with raises(
        TypeError,
        match=r".* elements should be \(<class 'int'>, <class 'numpy.integer'>, <class 'float'>, <class 'float'>\).*",
    ):
        spline_fxns.splev_deg0(x, xi, i)

    xi = [0, 1, 0, 3]
    with raises(ValueError, match="xi must be a non-decreasing sequence"):
        spline_fxns.splev_deg0(x, xi, i)

    xi = [0, 1, 2, 3]
    with raises(ValueError, match="x cannot be empty"):
        spline_fxns.splev_deg0(x, xi, i)

    x = ["a", "b", "c"]
    with raises(
        TypeError,
        match=r".* elements should be \(<class 'int'>, <class 'numpy.integer'>, <class 'float'>, <class 'float'>\).*",
    ):
        spline_fxns.splev_deg0(x, xi, i)

    x = np.array([-0.5, 0, 0.5, 1])
    xi = np.array([-1, 0, 1, 2])
    b0 = spline_fxns.splev_deg0(x, xi, i)
    assert (b0 == np.array([0, 1, 1, 0])).all()

    xi = np.array([-1, 0, 1])
    b0 = spline_fxns.splev_deg0(x, xi, i)
    assert (b0 == np.array([0, 1, 1, 1])).all()


def test_splev_degreecontrol():
    x = []
    xi = []
    cs = []
    p = 1.5
    tck = tuple([xi, cs, p])
    with raises(
        TypeError, match=r".* should be \(<class 'int'>, <class 'numpy.integer'>\).*"
    ):
        spline_fxns.splev_degreecontrol(x, tck)

    p = -1
    tck = tuple([xi, cs, p])
    with raises(ValueError, match=r"tck\[1\] cannot be empty"):
        spline_fxns.splev_degreecontrol(x, tck)

    cs = [[0, 1], [1, 1], [2, 0]]
    tck = tuple([xi, cs, p])
    with raises(
        TypeError,
        match=r".* elements should be \(<class 'int'>, <class 'numpy.integer'>, <class 'float'>, <class 'float'>\).*",
    ):
        spline_fxns.splev_degreecontrol(x, tck)

    cs = [0, 1, 2]
    tck = tuple([xi, cs, p])
    with raises(ValueError, match=r"tck\[0\] cannot be empty"):
        spline_fxns.splev_degreecontrol(x, tck)

    xi = ["a"]
    tck = tuple([xi, cs, p])
    with raises(
        TypeError,
        match=r".* elements should be \(<class 'int'>, <class 'numpy.integer'>, <class 'float'>, <class 'float'>\).*",
    ):
        spline_fxns.splev_degreecontrol(x, tck)

    xi = [0, 2, 1]
    tck = tuple([xi, cs, p])
    with raises(ValueError, match=r"tck\[0\] must be a non-decreasing sequence"):
        spline_fxns.splev_degreecontrol(x, tck)

    xi = [0, 1, 2]
    tck = tuple([xi, cs, p])
    with raises(ValueError, match=r"x cannot be empty"):
        spline_fxns.splev_degreecontrol(x, tck)

    x = ["a"]
    tck = tuple([xi, cs, p])
    with raises(
        TypeError,
        match=r".* elements should be \(<class 'int'>, <class 'numpy.integer'>, <class 'float'>, <class 'float'>\).*",
    ):
        spline_fxns.splev_degreecontrol(x, tck)

    # TEST p < 0
    x = np.linspace(-1, 1, 4)
    p = -1
    tck = tuple([xi, cs, p])
    b = spline_fxns.splev_degreecontrol(x, tck)
    assert (b == np.zeros(len(x))).all()

    # TEST p = 0
    xi = np.array([0, 0.5, 1])
    cs = np.array([1, 1])
    p = 0
    tck = tuple([xi, cs, p])
    b = spline_fxns.splev_degreecontrol(x, tck)
    assert (b == np.array([0, 0, 1, 1])).all()

    # TEST p > 0
    xx = np.linspace(0, 2 * np.pi)
    X, Y = xx, np.sin(xx)
    tck, u = splprep([X, Y])
    expected_newpoints = splev(u, tck)
    cs = tck[1]
    newpoints = []
    for coord in cs:
        coord_tck = tuple([tck[0], coord, tck[2]])
        newpoints.append(spline_fxns.splev_degreecontrol(u, coord_tck))
    newpoints = np.stack(newpoints, axis=1)
    new_x = newpoints[:, 0]
    new_y = newpoints[:, 1]
    assert (new_x == approx(expected_newpoints[0]) and new_y == approx(expected_newpoints[1]))

