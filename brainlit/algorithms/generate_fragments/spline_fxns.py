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


# def speed(x, tck):
#     derivs = []
#     for coord in tck[1]:
#         tck_coord = list(tck)
#         tck_coord[1] = coord
#         tck_coord = tuple(tck_coord)

#         derivs.append(splev_deriv(x, tck_coord))

#     deriv = np.stack(derivs, axis=1)

#     speed = np.linalg.norm(deriv, axis=1)
#     return speed


def curvature(x: np.ndarray, t: np.ndarray, c: np.ndarray, k: np.integer) -> np.ndarray:
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
    # transpose deriv, dderiv
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

    return curvature


# def torsion(x, tck):
#     derivs = []
#     deriv2s = []
#     deriv3s = []
#     for coord in tck[1]:
#         tck_coord = list(tck)
#         tck_coord[1] = coord
#         tck_coord = tuple(tck_coord)

#         derivs.append(splev_deriv(x, tck_coord))
#         deriv2s.append(splev_deriv2(x, tck_coord))
#         deriv3s.append(splev_deriv3(x, tck_coord))

#     deriv = np.stack(derivs, axis=1)
#     deriv2 = np.stack(deriv2s, axis=1)
#     deriv3 = np.stack(deriv3s, axis=1)

#     cross = np.cross(deriv, deriv2)

#     # Could be more efficient by only computing dot products of corresponding rows
#     num = np.diag((cross @ deriv3.T))
#     denom = np.linalg.norm(cross, axis=1) ** 2

#     torsion = np.nan_to_num(num / denom)

#     return torsion
