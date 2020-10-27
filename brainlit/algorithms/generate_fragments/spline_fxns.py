import operator
from scipy.interpolate import BSpline, splev
import numpy as np
from typing import Iterable, Union, Tuple
from brainlit.utils.util import (
    check_type,
    check_size,
    check_precomputed,
    check_iterable_type,
    check_iterable_nonnegative,
)


def speed(
    x: np.ndarray,
    t: np.ndarray,
    c: np.ndarray,
    k: np.integer,
    aux_outputs: bool = False,
) -> np.ndarray:
    r"""Compute the speed of a B-Spline.

    The speed is the norm of the first derivative of the B-Spline.

    Arguments:
        x: A `1xL` array of parameter values where to evaluate the curve.
            It contains the parameter values where the speed of the B-Spline will
            be evaluated. It is required to be non-empty, one-dimensional, and
            real-valued.
        t: A `1xm` array representing the knots of the B-spline.
            It is required to be a non-empty, non-decreasing, and one-dimensional
            sequence of real-valued elements. For a B-Spline of degree `k`, at least
            `2k + 1` knots are required.
        c: A `dxn` array representing the coefficients/control points of the B-spline.
            Given `n` real-valued, `d`-dimensional points ::math::`x_k = (x_k(1),...,x_k(d))`,
            `c` is the non-empty matrix which columns are ::math::`x_1^T,...,x_N^T`. For a
            B-Spline of order `k`, `n` cannot be less than `m-k-1`.
        k: A non-negative integer representing the degree of the B-spline.

    Returns:
        speed: A `1xL` array containing the speed of the B-Spline evaluated at `x`

    References:
    .. [1] Kouba, Parametric Equations.
        https://www.math.ucdavis.edu/~kouba/Math21BHWDIRECTORY/ArcLength.pdf
    """

    # convert arguments to desired type
    x = np.ascontiguousarray(x)
    t = np.ascontiguousarray(t)
    c = np.ascontiguousarray(c)
    k = operator.index(k)

    if k < 0:
        raise ValueError("The order of the spline must be non-negative")

    check_type(t, np.ndarray)
    t_dim = t.ndim
    if t_dim != 1:
        raise ValueError("t must be one-dimensional")
    if len(t) == 0:
        raise ValueError("t must be non-empty")
    check_iterable_type(t, (np.integer, np.float))
    if (np.diff(t) < 0).any():
        raise ValueError("t must be a non-decreasing sequence")

    check_type(c, np.ndarray)
    c_dim = c.ndim
    if c_dim > 2:
        raise ValueError("c must be 2D max")
    if len(c.flatten()) == 0:
        raise ValueError("c must be non-empty")
    if c_dim == 1:
        check_iterable_type(c, (np.integer, np.float))
        # expand dims so that we can cycle through a single dimension
        c = np.expand_dims(c, axis=0)
    if c_dim == 2:
        for d in c:
            check_iterable_type(d, (np.integer, np.float))
    n_dim = len(c)

    check_type(x, np.ndarray)
    x_dim = x.ndim
    if x_dim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    check_iterable_type(x, (np.integer, np.float))
    L = len(x)

    # evaluate first and second derivatives
    # deriv, dderiv are (d, L) arrays
    deriv = np.empty((n_dim, L))
    for i, dim in enumerate(c):
        spl = BSpline(t, dim, k)
        deriv[i, :] = spl.derivative(nu=1)(x) if k - 1 >= 0 else np.zeros(L)
    # tranpose deriv
    deriv = deriv.T

    speed = np.linalg.norm(deriv, axis=1)
    if aux_outputs == False:
        return speed
    else:
        return speed, deriv


def curvature(
    x: np.ndarray,
    t: np.ndarray,
    c: np.ndarray,
    k: np.integer,
    aux_outputs: bool = False,
) -> np.ndarray:
    r"""Compute the curvature of a B-Spline.

    The curvature measures the failure of a curve, `r(u)`, to be a straight line.
    It is defined as

    .. math::

        k = \lVert \frac{dT}{ds} \rVert,

    where `T` is the unit tangent vector, and `s` is the arc length:

    .. math::

        T = \frac{dr}{ds},\quad s = \int_0^t \lVert r'(u) \rVert du,

    where `r(u)` is the position vector as a function of time.

    The curvature can also be computed as

    .. math::

        k = \lVert r'(t) \times r''(t)\rVert / \lVert r'(t) \rVert^3.

    Arguments:
        x: A `1xL` array of parameter values where to evaluate the curve.
            It contains the parameter values where the curvature of the B-Spline will
            be evaluated. It is required to be non-empty, one-dimensional, and
            real-valued.
        t: A `1xm` array representing the knots of the B-spline.
            It is required to be a non-empty, non-decreasing, and one-dimensional
            sequence of real-valued elements. For a B-Spline of degree `k`, at least
            `2k + 1` knots are required.
        c: A `dxn` array representing the coefficients/control points of the B-spline.
            Given `n` real-valued, `d`-dimensional points ::math::`x_k = (x_k(1),...,x_k(d))`,
            `c` is the non-empty matrix which columns are ::math::`x_1^T,...,x_N^T`. For a
            B-Spline of order `k`, `n` cannot be less than `m-k-1`.
        k: A non-negative integer representing the degree of the B-spline.

    Returns:
        curvature: A `1xL` array containing the curvature of the B-Spline evaluated at `x`

    References:
    .. [1] Máté Attila, The Frenet–Serret formulas.
        http://www.sci.brooklyn.cuny.edu/~mate/misc/frenet_serret.pdf
    """

    # convert arguments to desired type
    x = np.ascontiguousarray(x)
    t = np.ascontiguousarray(t)
    c = np.ascontiguousarray(c)
    k = operator.index(k)

    if k < 0:
        raise ValueError("The order of the spline must be non-negative")

    check_type(t, np.ndarray)
    t_dim = t.ndim
    if t_dim != 1:
        raise ValueError("t must be one-dimensional")
    if len(t) == 0:
        raise ValueError("t must be non-empty")
    check_iterable_type(t, (np.integer, np.float))
    if (np.diff(t) < 0).any():
        raise ValueError("t must be a non-decreasing sequence")

    check_type(c, np.ndarray)
    c_dim = c.ndim
    if c_dim > 2:
        raise ValueError("c must be 2D max")
    if len(c.flatten()) == 0:
        raise ValueError("c must be non-empty")
    if c_dim == 1:
        check_iterable_type(c, (np.integer, np.float))
        # expand dims so that we can cycle through a single dimension
        c = np.expand_dims(c, axis=0)
    if c_dim == 2:
        for d in c:
            check_iterable_type(d, (np.integer, np.float))
    n_dim = len(c)

    check_type(x, np.ndarray)
    x_dim = x.ndim
    if x_dim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    check_iterable_type(x, (np.integer, np.float))
    L = len(x)

    # evaluate first and second derivatives
    # deriv, dderiv are (d, L) arrays
    deriv = np.empty((n_dim, L))
    dderiv = np.empty((n_dim, L))
    for i, dim in enumerate(c):
        spl = BSpline(t, dim, k)
        deriv[i, :] = spl.derivative(nu=1)(x) if k - 1 >= 0 else np.zeros(L)
        dderiv[i, :] = spl.derivative(nu=2)(x) if k - 2 >= 0 else np.zeros(L)
    # transpose derivs
    deriv = deriv.T
    dderiv = dderiv.T
    # evaluate the cross product
    cross = np.cross(deriv, dderiv)
    # evalute the curvature
    num = np.linalg.norm(cross, axis=1)
    denom = np.linalg.norm(deriv, axis=1) ** 3
    curvature = np.nan_to_num(num / denom)

    if aux_outputs == True:
        return curvature, deriv, dderiv
    else:
        return curvature


def torsion(
    x: np.ndarray,
    t: np.ndarray,
    c: np.ndarray,
    k: np.integer,
    aux_outputs: bool = False,
) -> np.ndarray:
    r"""Compute the torsion of a B-Spline.

    The torsion measures the failure of a curve, `r(u)`, to be planar.
    If the curvature `k` of a curve is not zero, then the torsion is defined as

    .. math::

        \tau = -n \cdot b',

    where `n` is the principal normal vector, and `b'` the derivative w.r.t. the
    arc length `s` of the binormal vector.

    The torsion can also be computed as

    .. math::
        \tau = \lvert r'(t), r''(t), r'''(t) \rvert / \lVert r'(t) \times r''(t) \rVert^2,

    where `r(u)` is the position vector as a function of time.

    Arguments:
        x: A `1xL` array of parameter values where to evaluate the curve.
            It contains the parameter values where the torsion of the B-Spline will
            be evaluated. It is required to be non-empty, one-dimensional, and
            real-valued.
        t: A `1xm` array representing the knots of the B-spline.
            It is required to be a non-empty, non-decreasing, and one-dimensional
            sequence of real-valued elements. For a B-Spline of degree `k`, at least
            `2k + 1` knots are required.
        c: A `dxn` array representing the coefficients/control points of the B-spline.
            Given `n` real-valued, `d`-dimensional points ::math::`x_k = (x_k(1),...,x_k(d))`,
            `c` is the non-empty matrix which columns are ::math::`x_1^T,...,x_N^T`. For a
            B-Spline of order `k`, `n` cannot be less than `m-k-1`.
        k: A non-negative integer representing the degree of the B-spline.

    Returns:
        torsion: A `1xL` array containing the torsion of the B-Spline evaluated at `x`

    References:
    .. [1] Máté Attila, The Frenet–Serret formulas.
        http://www.sci.brooklyn.cuny.edu/~mate/misc/frenet_serret.pdf
    """

    # convert arguments to desired type
    x = np.ascontiguousarray(x)
    t = np.ascontiguousarray(t)
    c = np.ascontiguousarray(c)
    k = operator.index(k)

    if k < 0:
        raise ValueError("The order of the spline must be non-negative")

    check_type(t, np.ndarray)
    t_dim = t.ndim
    if t_dim != 1:
        raise ValueError("t must be one-dimensional")
    if len(t) == 0:
        raise ValueError("t must be non-empty")
    check_iterable_type(t, (np.integer, np.float))
    if (np.diff(t) < 0).any():
        raise ValueError("t must be a non-decreasing sequence")

    check_type(c, np.ndarray)
    c_dim = c.ndim
    if c_dim > 2:
        raise ValueError("c must be 2D max")
    if len(c.flatten()) == 0:
        raise ValueError("c must be non-empty")
    if c_dim == 1:
        check_iterable_type(c, (np.integer, np.float))
        # expand dims so that we can cycle through a single dimension
        c = np.expand_dims(c, axis=0)
    if c_dim == 2:
        for d in c:
            check_iterable_type(d, (np.integer, np.float))
    n_dim = len(c)

    check_type(x, np.ndarray)
    x_dim = x.ndim
    if x_dim != 1:
        raise ValueError("x must be one-dimensional")
    if len(x) == 0:
        raise ValueError("x must be non-empty")
    check_iterable_type(x, (np.integer, np.float))
    L = len(x)

    # evaluate first, second, and third derivatives
    # deriv, dderiv, ddderiv are (d, L) arrays
    deriv = np.empty((n_dim, L))
    dderiv = np.empty((n_dim, L))
    ddderiv = np.empty((n_dim, L))
    for i, dim in enumerate(c):
        spl = BSpline(t, dim, k)
        deriv[i, :] = spl.derivative(nu=1)(x) if k - 1 >= 0 else np.zeros(L)
        dderiv[i, :] = spl.derivative(nu=2)(x) if k - 2 >= 0 else np.zeros(L)
        ddderiv[i, :] = spl.derivative(nu=3)(x) if k - 3 >= 0 else np.zeros(L)
    # transpose derivs
    deriv = deriv.T
    dderiv = dderiv.T
    ddderiv = ddderiv.T

    cross = np.cross(deriv, dderiv)

    # Could be more efficient by only computing dot products of corresponding rows
    num = np.diag((cross @ ddderiv.T))
    denom = np.linalg.norm(cross, axis=1) ** 2

    torsion = np.nan_to_num(num / denom)

    if aux_outputs == True:
        return torsion, deriv, dderiv, ddderiv
    else:
        return torsion
