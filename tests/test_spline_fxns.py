import numpy as np
from brainlit.algorithms.generate_fragments import spline_fxns
from pytest import approx, raises
from scipy.interpolate import splprep, splev, BSpline


def test_curvature():
    x = [[]]
    t = [[]]
    c = [[[]]]
    k = 1.5

    # test k must be an integer
    with raises(TypeError, match=r"'float' object cannot be interpreted as an integer"):
        spline_fxns.curvature(x, t, c, k)
    # test k must be non-negative
    k = -1
    with raises(ValueError, match=r"The order of the spline must be non-negative"):
        spline_fxns.curvature(x, t, c, k)
    # test t must be one-dimensional
    k = 0
    with raises(ValueError, match=r"t must be one-dimensional"):
        spline_fxns.curvature(x, t, c, k)
    # test t must be non-empty
    t = []
    with raises(ValueError, match=r"t must be non-empty"):
        spline_fxns.curvature(x, t, c, k)
    # test t must contain integers or floats
    t = ["abc"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.curvature(x, t, c, k)
    # test t must be non-decreasing
    t = [2, 1, 0]
    with raises(ValueError, match=r"t must be a non-decreasing sequence"):
        spline_fxns.curvature(x, t, c, k)
    # test c must be no more than 2-dimensional
    t = [0, 1, 2]
    with raises(ValueError, match=r"c must be 2D max"):
        spline_fxns.curvature(x, t, c, k)
    # test c must be non-empty
    c = [[]]
    with raises(ValueError, match=r"c must be non-empty"):
        spline_fxns.curvature(x, t, c, k)
    # test c must contain integers or floats (1D case)
    c = ["acb"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.curvature(x, t, c, k)
    # test c must contain integers or floats (ND case)
    c = [["abc"], ["def"], ["ghi"]]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.curvature(x, t, c, k)
    # test x must be one-dimensional
    c = [1, 2, 3, 4]
    with raises(ValueError, match=r"x must be one-dimensional"):
        spline_fxns.curvature(x, t, c, k)
    # test x must be non-empty
    x = []
    with raises(ValueError, match=r"x must be non-empty"):
        spline_fxns.curvature(x, t, c, k)
    # test x must contain integers or floats
    x = ["abc"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.curvature(x, t, c, k)

    # test curvature is 0 when C is a straight line
    xx = np.linspace(-1, 1, 10)
    X = xx
    Y = X
    Z = X
    dX = np.ones(len(xx))
    dY = dX
    dZ = dX
    ddX = np.zeros(len(xx))
    ddY = ddX
    ddZ = ddY

    C = [X, Y, Z]

    tck, u = splprep(C, u=xx, k=2)
    t = tck[0]
    c = tck[1]
    k = tck[2]

    curvature = spline_fxns.curvature(u, t, c, k)
    assert (curvature == np.zeros(len(xx))).all()

    # test results in general case
    xx = np.linspace(-np.pi, np.pi)
    X = xx ** 3
    Y = np.sin(xx)
    Z = xx ** 2
    dX = 3 * xx ** 2
    dY = np.cos(xx)
    dZ = 2 * xx
    ddX = 6 * xx
    ddY = -np.sin(xx)
    ddZ = 2 * np.ones(len(xx))

    C = [X, Y, Z]
    dC = [dX, dY, dZ]
    ddC = [ddX, ddY, ddZ]

    tck, u = splprep(C, u=xx, k=5)
    t = tck[0]
    c = tck[1]
    k = tck[2]
    curvature = spline_fxns.curvature(xx, t, c, k, aux_outputs=False)

    expected_deriv = np.array(dC).T
    expected_dderiv = np.array(ddC).T
    expected_cross = np.cross(expected_deriv, expected_dderiv)
    num = np.linalg.norm(expected_cross, axis=1)
    denom = np.linalg.norm(expected_deriv, axis=1) ** 3
    expected_curvature = np.nan_to_num(num / denom)

    assert curvature == approx(expected_curvature, abs=1e-1)


def test_torsion():
    x = [[]]
    t = [[]]
    c = [[[]]]
    k = 1.5

    # test k must be an integer
    with raises(TypeError, match=r"'float' object cannot be interpreted as an integer"):
        spline_fxns.torsion(x, t, c, k)
    # test k must be non-negative
    k = -1
    with raises(ValueError, match=r"The order of the spline must be non-negative"):
        spline_fxns.torsion(x, t, c, k)
    # test t must be one-dimensional
    k = 0
    with raises(ValueError, match=r"t must be one-dimensional"):
        spline_fxns.torsion(x, t, c, k)
    # test t must be non-empty
    t = []
    with raises(ValueError, match=r"t must be non-empty"):
        spline_fxns.torsion(x, t, c, k)
    # test t must contain integers or floats
    t = ["abc"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.torsion(x, t, c, k)
    # test t must be non-decreasing
    t = [2, 1, 0]
    with raises(ValueError, match=r"t must be a non-decreasing sequence"):
        spline_fxns.torsion(x, t, c, k)
    # test c must be no more than 2-dimensional
    t = [0, 1, 2]
    with raises(ValueError, match=r"c must be 2D max"):
        spline_fxns.torsion(x, t, c, k)
    # test c must be non-empty
    c = [[]]
    with raises(ValueError, match=r"c must be non-empty"):
        spline_fxns.torsion(x, t, c, k)
    # test c must contain integers or floats (1D case)
    c = ["acb"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.torsion(x, t, c, k)
    # test c must contain integers or floats (ND case)
    c = [["abc"], ["def"], ["ghi"]]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.torsion(x, t, c, k)
    # test x must be one-dimensional
    c = [1, 2, 3, 4]
    with raises(ValueError, match=r"x must be one-dimensional"):
        spline_fxns.torsion(x, t, c, k)
    # test x must be non-empty
    x = []
    with raises(ValueError, match=r"x must be non-empty"):
        spline_fxns.torsion(x, t, c, k)
    # test x must contain integers or floats
    x = ["abc"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.torsion(x, t, c, k)

    # test results in general case
    xx = np.linspace(-np.pi, np.pi)
    X = xx ** 3
    Y = np.sin(xx)
    Z = xx ** 2
    dX = 3 * xx ** 2
    dY = np.cos(xx)
    dZ = 2 * xx
    ddX = 6 * xx
    ddY = -np.sin(xx)
    ddZ = 2 * np.ones(len(xx))
    dddX = 6 * np.ones(len(xx))
    dddY = -np.cos(xx)
    dddZ = np.zeros(len(xx))

    C = np.array([X, Y, Z])
    dC = np.array([dX, dY, dZ]).T
    ddC = np.array([ddX, ddY, ddZ]).T
    dddC = np.array([dddX, dddY, dddZ]).T

    tck, _ = splprep(C, u=xx, k=5)
    t = tck[0]
    c = tck[1]
    k = tck[2]
    torsion = spline_fxns.torsion(xx, t, c, k, aux_outputs=False)

    expected_cross = np.cross(dC, ddC)
    expected_num = np.diag((expected_cross @ dddC.T))
    expected_denom = np.linalg.norm(expected_cross, axis=1) ** 2
    expected_torsion = np.nan_to_num(expected_num / expected_denom)

    assert torsion == approx(expected_torsion, abs=1e-1)


def test_speed():
    x = [[]]
    t = [[]]
    c = [[[]]]
    k = 1.5

    # test k must be an integer
    with raises(TypeError, match=r"'float' object cannot be interpreted as an integer"):
        spline_fxns.speed(x, t, c, k)
    # test k must be non-negative
    k = -1
    with raises(ValueError, match=r"The order of the spline must be non-negative"):
        spline_fxns.speed(x, t, c, k)
    # test t must be one-dimensional
    k = 0
    with raises(ValueError, match=r"t must be one-dimensional"):
        spline_fxns.speed(x, t, c, k)
    # test t must be non-empty
    t = []
    with raises(ValueError, match=r"t must be non-empty"):
        spline_fxns.speed(x, t, c, k)
    # test t must contain integers or floats
    t = ["abc"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.speed(x, t, c, k)
    # test t must be non-decreasing
    t = [2, 1, 0]
    with raises(ValueError, match=r"t must be a non-decreasing sequence"):
        spline_fxns.speed(x, t, c, k)
    # test c must be no more than 2-dimensional
    t = [0, 1, 2]
    with raises(ValueError, match=r"c must be 2D max"):
        spline_fxns.speed(x, t, c, k)
    # test c must be non-empty
    c = [[]]
    with raises(ValueError, match=r"c must be non-empty"):
        spline_fxns.speed(x, t, c, k)
    # test c must contain integers or floats (1D case)
    c = ["acb"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.speed(x, t, c, k)
    # test c must contain integers or floats (ND case)
    c = [["abc"], ["def"], ["ghi"]]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.speed(x, t, c, k)
    # test x must be one-dimensional
    c = [1, 2, 3, 4]
    with raises(ValueError, match=r"x must be one-dimensional"):
        spline_fxns.speed(x, t, c, k)
    # test x must be non-empty
    x = []
    with raises(ValueError, match=r"x must be non-empty"):
        spline_fxns.speed(x, t, c, k)
    # test x must contain integers or floats
    x = ["abc"]
    with raises(
        TypeError,
        match=r"elements should be \(<class 'numpy.integer'>, <class 'float'>\)",
    ):
        spline_fxns.speed(x, t, c, k)

    # test results in general case
    # test results in general case
    xx = np.linspace(-np.pi, np.pi)
    X = xx ** 3
    Y = np.sin(xx)
    Z = xx ** 2
    dX = 3 * xx ** 2
    dY = np.cos(xx)
    dZ = 2 * xx

    C = np.array([X, Y, Z])
    dC = np.array([dX, dY, dZ]).T

    tck, _ = splprep(C, u=xx, k=5)
    t = tck[0]
    c = tck[1]
    k = tck[2]
    speed = spline_fxns.speed(xx, t, c, k, aux_outputs=False)

    expected_speed = np.linalg.norm(dC, axis=1)

    assert speed == approx(expected_speed, abs=1e-1)
