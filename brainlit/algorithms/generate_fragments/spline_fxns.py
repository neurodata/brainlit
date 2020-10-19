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
    """Compute the speed of a B-Spline.

    The speed is the norm of the first derivative of the B-Spline

    Arguments:
        x: A 1xL array of parameter values where to evaluate the curve
        t: A 1xm array representing the knots of the B-spline
        c: A dxn array representing the control points of the B-spline
        k: An integer representing the degree of the B-spline:

    Returns:
        speed: A 1xL array containing the speed of the B-Spline evaluated at x
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
    """Compute the curvature of a B-Spline.

    The curvature measures the failure of a curve to be a straight line.
    It is defined as

    k = ||dT/ds||,

    where T is the unit tangent vector, and s is the arc length:

    T = dr/ds, s = int_0^t ||r'(u)||du.

    The curvature can also be computed as

    k = ||r'(t) x r''(t)||/||r'(t)||^3.

    Arguments:
        x: A 1xL array of parameter values where to evaluate the curve
        t: A 1xm array representing the knots of the B-spline
        c: A dxn array representing the control points of the B-spline
        k: An integer representing the degree of the B-spline:

    Returns:
        curvature: A 1xL array containing the curvature of the B-Spline evaluated at x
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

    if np.isnan(np.sum(curvature)):
        print("torsion nan")

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
    """Compute the torsion of a B-Spline.

    The torsion measures the failure of a line to be planar.
    If the curvature k of a curve is not zero, then the torsion is defined as

    \tau = -n * b',

    where n is the principal normal vector, and b' the derivative w.r.t. the
    arc length s of the binormal vector.

    The torsion can also be computed as

    \tau = |r'(t), r''(t), r'''(t)|/||r'(t) x r''(t)||^2.

    Arguments:
        x: A 1xL array of parameter values where to evaluate the curve
        t: A 1xm array representing the knots of the B-spline
        c: A dxn array representing the control points of the B-spline
        k: An integer representing the degree of the B-spline:

    Returns:
        torsion: A 1xL array containing the torsion of the B-Spline evaluated at x
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
