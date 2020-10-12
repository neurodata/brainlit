from scipy.interpolate import splev
import numpy as np
from typing import Iterable, Union, Tuple
from brainlit.utils.util import (
    check_type,
    check_size,
    check_precomputed,
    check_iterable_type,
    check_iterable_nonnegative,
)


def splev_deg0(x: np.ndarray, xi: np.ndarray, i: int) -> np.ndarray:
    """Compute the i-th B-spline of degree 0.
    
    The i-th B-spline of degree 0 is defined as the characteristic function 
    of the half open interval [xi[i], x[i+1]). So:
    
    b[i,0,xi](x) = 1 if x[i] <= x < x[i+1], 0 otherwise

    Arguments:
        x: A 1xn array of parameter values where to evaluate the B-spline
        xi: The 1xm array representing the knot sequence of the B-spline
        i: Index of the B-spline to compute
    Returns:
        within: A 1xn binary array representing the i-th degree-0 B-spline
    """
    # Check that i is an integer
    check_type(i, (int, np.integer))
    # Check that xi is a non-empty, non-decreasing sequence of floats or ints
    check_iterable_type(xi, (int, np.integer, float, np.float))
    m = len(xi)
    if m == 0:
        raise ValueError(("xi cannot be empty"))
    non_decreasing = [x <= xi[i + 1] for i, x in enumerate(xi[:-1])]
    if not all(non_decreasing):
        raise ValueError(("xi must be a non-decreasing sequence"))
    # Check that x is a non-empty sequence of floats or ints
    check_iterable_type(x, (int, np.integer, float, np.float))
    n = len(x)
    if n == 0:
        raise ValueError(("x cannot be empty"))

    if i < m - 2:
        within = (x >= xi[i]) & (x < xi[i + 1])
    else:
        within = (x >= xi[i]) & (x <= xi[i + 1])
    return np.array(1 * (within))


def splev_degreecontrol(
    x: np.ndarray, tck: Tuple[np.ndarray, np.ndarray, Union[int, np.integer]]
) -> np.ndarray:
    """Evaluate the curve S(x) from a B-spline.
    
    S(x) = (s[1], ..., s[L]), s[k] = sum_{i=0}^n cs[i] * b[i, p](x[k]), where:
        * cs is the array of control points
        * n is the length of cs
        * b[i, p](x[k]) is the i-th B-spline of degree p evaluated at x[k]

    Arguments:
        x: A 1xL array of parameter values where to evaluate the curve
        tck: A three-elements tuple where:
            * tck[0]: A 1xm array representing the knots of the B-spline
            * tck[1]: A 1xn array representing the control points of the B-spline
            * tck[2]: An integer representing the degree of the B-spline
    Returns:
        A 1xL array representing the curve S(x)
    """
    # check that p = tck[2] aka the degree of the B-spline is an integer
    check_type(tck[2], (int, np.integer))
    p = tck[2]
    # check that cs = tck[1] aka the control points of the B-spline are a non-empty flat array
    check_iterable_type(tck[1], (int, np.integer, float, np.float))
    cs = tck[1]
    n = len(cs)
    if n == 0:
        raise ValueError(("tck[1] cannot be empty"))
    # Check that xi = tck[0] aka the knots are a non-empty, non-decreasing sequence of floats or ints
    check_iterable_type(tck[0], (int, np.integer, float, np.float))
    xi = tck[0]
    m = len(xi)
    if m == 0:
        raise ValueError(("tck[0] cannot be empty"))
    non_decreasing = [x <= xi[i + 1] for i, x in enumerate(xi[:-1])]
    if not all(non_decreasing):
        raise ValueError(("tck[0] must be a non-decreasing sequence"))
    # check that x is a non-empty sequence
    check_iterable_type(x, (int, np.integer, float, np.float))
    L = len(x)
    if L == 0:
        raise ValueError(("x cannot be empty"))

    if p < 0:
        return 0 * x
    elif p == 0:
        val = 0 * x
        for j, c in enumerate(cs):
            if c != 0:
                val = val + c * splev_deg0(x, xi, j)
        return val
    else:
        return splev(x, tck)


def splev_deriv(x, tck):
    """Evaluate the first derivative S'(x) of a B-spline.

    S'(x) = (s[1], ..., s[L]), s[k] = sum_{j=0}^n cs[j] * p * (A - B), p >= 1,
    where:
    A = b[j, p-1, xi](x) / (xi[j+p] - xi[j])
    B = b[j+1, p-1, xi](x) / (xi[j+p+1] - xi[j+i])

    Arguments:
        x: A 1xL array of parameter values where to evaluate the curve
        tck: A three-elements tuple where:
            * tck[0]: A 1xm array representing the knots of the B-spline
            * tck[1]: A 1xn array representing the control points of the B-spline
            * tck[2]: An integer representing the degree of the B-spline
    Returns:
    """
    # check that p = tck[2] aka the degree of the B-spline is an integer
    check_type(tck[2], (int, np.integer))
    p = tck[2]
    # check that cs = tck[1] aka the control points of the B-spline are a non-empty flat array
    check_iterable_type(tck[1], (int, np.integer, float, np.float))
    cs = tck[1]
    n = len(cs)
    if n == 0:
        raise ValueError(("tck[1] cannot be empty"))
    cs = tck[1]
    # Check that xi = tck[0] aka the knots are a non-empty, non-decreasing sequence of floats or ints
    check_iterable_type(tck[0], (int, np.integer, float, np.float))
    xi = tck[0]
    m = len(xi)
    if m == 0:
        raise ValueError(("tck[0] cannot be empty"))
    non_decreasing = [x <= xi[i + 1] for i, x in enumerate(xi[:-1])]
    if not all(non_decreasing):
        raise ValueError(("tck[0] must be a non-decreasing sequence"))
    # check that x is a non-empty sequence
    check_iterable_type(x, (int, np.integer, float, np.float))
    L = len(x)
    if L == 0:
        raise ValueError(("x cannot be empty"))

    # define p+1 extra knots
    padding = np.ones(p + 1)
    pre_xi = (xi[0] - 1) * padding
    post_xi = (xi[-1] + 1) * padding
    # define new xi such that
    # xi_new[0] == ... == xi_new[p] := xi_old[0] - 1
    # xi_new[p+1] <= ... <= xi_new[m+p] := xi_old
    # xi_new[m+p+1] == ... == xi[m+2p+1] := xi_old[-1] + 1
    xi = np.concatenate((pre_xi, xi, post_xi))
    # define new cs such that
    # cs_new[0] == ... == cs_new[p] := 0
    # cs_new[p+1] <= ... <= cs_new[n+p] := cs_old
    # cs_new[n+p+1] == ... == xi[n+2p+1] := 0
    cs = np.concatenate((np.zeros(p + 1), cs, np.zeros(p + 1)))
    # n_new = n_old + 2*p + 1
    n = len(cs)
    # initialize the derivative array
    L = len(x)
    deriv = np.zeros(L)

    # j cycles through the original points of cs_old
    # j = [p+1, p+1, ..., n_old+p]
    for j in np.arange(p + 1, n - (p + 1)):

        # compute A = b[j, p-1, xi](x) / (xi[j+p] - xi[j])
        xi_jp = xi[j + p]
        xi_j = xi[j]
        c1 = 0 if xi_j == xi_jp else 1 / (xi_jp - xi_j)

        tckb1 = list(tck)
        cb1 = 0 * cs
        cb1[j] = 1
        tckb1[0] = xi
        tckb1[1] = cb1
        tckb1[2] = p - 1
        tckb1 = tuple(tckb1)

        A = splev_degreecontrol(x, tckb1) * c1

        # compute B = b[j+1, p-1, xi](x) / (xi[j+p+1] - xi[j+i])
        xi_jp1 = xi[j + p + 1]
        xi_j1 = xi[j + 1]
        c2 = 0 if xi_j1 == xi_jp1 else 1 / (xi_jp1 - xi_j1)

        tckb2 = list(tck)
        cb2 = 0 * cs
        cb2[j + 1] = 1
        tckb2[0] = xi
        tckb2[1] = cb2
        tckb2[2] = p - 1
        tckb2 = tuple(tckb2)

        B = splev_degreecontrol(x, tckb2) * c2

        deriv = deriv + cs[j] * (A - B)
    deriv = deriv * p

    return deriv


def splev_deriv2(x, tck):
    cs = tck[1]
    p = tck[2]
    xi = tck[0]

    pre_xi = (xi[0] - 1) * np.ones((p + 1))
    post_xi = (xi[-1] + 1) * np.ones((p + 1))

    xi = np.concatenate((pre_xi, xi, post_xi))
    cs = np.concatenate((np.zeros(p + 1), cs, np.zeros(p + 1)))
    n = len(cs)

    deriv2 = np.zeros((len(x)))

    for j in np.arange(p + 1, n - p - 1):
        xi_jp = xi[j + p]
        xi_j = xi[j]
        if xi_j != xi_jp:
            c1 = 1 / (xi_jp - xi_j)
        else:
            c1 = 0

        xi_jp1 = xi[j + p + 1]
        xi_j1 = xi[j + 1]
        if xi_j1 != xi_jp1:
            c2 = 1 / (xi_jp1 - xi_j1)
        else:
            c2 = 0

        xi_jpm1 = xi[j + p - 1]
        if xi_j != xi_jpm1:
            c1a = 1 / (xi_jpm1 - xi_j)
        else:
            c1a = 0

        if xi_jp != xi_j1:
            c1b = 1 / (xi_jp - xi_j1)
        else:
            c1b = 0

        if xi_jp != xi_j1:
            c2a = 1 / (xi_jp - xi_j1)
        else:
            c2a = 0

        xi_j2 = xi[j + 2]
        if xi_jp1 != xi_j2:
            c2b = 1 / (xi_jp1 - xi_j2)
        else:
            c2b = 0

        cj = 0 * cs
        cj[j] = 1
        pm2 = p - 2
        tck1 = (xi, cj, pm2)
        d1 = splev_degreecontrol(x, tck1)

        cj1 = 0 * cs
        cj1[j + 1] = 1
        tck2 = (xi, cj1, pm2)
        d2 = splev_degreecontrol(x, tck2)

        cj2 = 0 * cs
        cj2[j + 2] = 1
        tck3 = (xi, cj2, pm2)
        d3 = splev_degreecontrol(x, tck3)

        deriv2 = deriv2 + cs[j] * (
            c1 * (c1a * d1 - c1b * d2) - c2 * (c2a * d2 - c2b * d3)
        )

    deriv2 = deriv2 * p * (p - 1)

    return deriv2


def splev_deriv3(x, tck):
    cs = tck[1]
    p = tck[2]
    xi = tck[0]

    pad = np.amax((3, p + 1))

    pre_xi = (xi[0] - 1) * np.ones(pad)
    post_xi = (xi[-1] + 1) * np.ones(pad)

    xi = np.concatenate((pre_xi, xi, post_xi))
    cs = np.concatenate((np.zeros(pad), cs, np.zeros(pad)))
    n = len(cs)

    deriv3 = np.zeros((len(x)))

    for j in np.arange(pad, n - pad):
        xi_jp = xi[j + p]
        xi_j = xi[j]
        if xi_j != xi_jp:
            c1 = 1 / (xi_jp - xi_j)
        else:
            c1 = 0

        xi_jp1 = xi[j + p + 1]
        xi_j1 = xi[j + 1]
        if xi_j1 != xi_jp1:
            c2 = 1 / (xi_jp1 - xi_j1)
        else:
            c2 = 0

        xi_jpm1 = xi[j + p - 1]
        if xi_j != xi_jpm1:
            c1a = 1 / (xi_jpm1 - xi_j)
        else:
            c1a = 0

        if xi_jp != xi_j1:
            c1b = 1 / (xi_jp - xi_j1)
        else:
            c1b = 0

        c2a = c1b

        xi_j2 = xi[j + 2]
        if xi_jp1 != xi_j2:
            c2b = 1 / (xi_jp1 - xi_j2)
        else:
            c2b = 0

        xi_jpm2 = xi[j + p - 2]
        if xi_jpm2 != xi_j:
            c1a1 = 1 / (xi_jpm2 - xi_j)
        else:
            c1a1 = 0

        if xi_jpm1 != xi_j1:
            c1a2 = 1 / (xi_jpm1 - xi_j1)
        else:
            c1a2 = 0

        c1b1 = c1a2
        xi_j2 = xi[j + 2]
        if xi_jp != xi_j2:
            c1b2 = 1 / (xi_jp - xi_j2)
        else:
            c1b2 = 0

        c2a1 = c1b1
        c2a2 = c1b2
        c2b1 = c1b2

        xi_j3 = xi[j + 3]
        if xi_jp1 != xi_j3:
            c2b2 = 1 / (xi_jp1 - xi_j3)
        else:
            c2b2 = 0

        # Utilize the 1st derivative function
        csj = 0 * cs
        csj[j] = 1
        pm3 = p - 3
        tck1a1 = (xi, csj, pm3)
        d1a1 = splev_degreecontrol(x, tck1a1)

        csj1 = 0 * cs
        csj1[j + 1] = 1
        tck1a2 = (xi, csj1, pm3)
        d1a2 = splev_degreecontrol(x, tck1a2)

        tck1b1 = (xi, csj1, pm3)
        d1b1 = splev_degreecontrol(x, tck1b1)

        csj2 = 0 * cs
        csj2[j + 2] = 1
        tck1b2 = (xi, csj2, pm3)
        d1b2 = splev_degreecontrol(x, tck1b2)

        d2a1 = d1b1
        d2a2 = d1b2

        tck2b1 = (xi, csj2, pm3)
        d2b1 = splev_degreecontrol(x, tck2b1)

        csj3 = 0 * cs
        csj3[j + 3] = 1
        tck2b2 = (xi, csj3, pm3)
        d2b2 = splev_degreecontrol(x, tck2b2)

        deriv3 = deriv3 + cs[j] * (
            c1 * (c1a * (d1a1 * c1a1 - d1a2 * c1a2) - c1b * (c1b1 * d1b1 - c1b2 * d1b2))
            - c2
            * (c2a * (c2a1 * d2a1 - c2a2 * d2a2) - c2b * (c2b1 * d2b1 - c2b2 * d2b2))
        )

    deriv3 = deriv3 * p * (p - 1) * (p - 2)

    return deriv3


def speed(x, tck):
    derivs = []
    for coord in tck[1]:
        tck_coord = list(tck)
        tck_coord[1] = coord
        tck_coord = tuple(tck_coord)

        derivs.append(splev_deriv(x, tck_coord))

    deriv = np.stack(derivs, axis=1)

    speed = np.linalg.norm(deriv, axis=1)
    return speed


def curvature(x, tck):
    derivs = []
    deriv2s = []
    for coord in tck[1]:
        tck_coord = list(tck)
        tck_coord[1] = coord
        tck_coord = tuple(tck_coord)

        derivs.append(splev_deriv(x, tck_coord))
        deriv2s.append(splev_deriv2(x, tck_coord))

    deriv = np.stack(derivs, axis=1)
    deriv2 = np.stack(deriv2s, axis=1)

    cross = np.cross(deriv, deriv2)

    num = np.linalg.norm(cross, axis=1)
    denom = np.linalg.norm(deriv, axis=1) ** 3

    curvature = np.nan_to_num(num / denom)
    if np.isnan(np.sum(curvature)):
        print("torsion nan")

    return curvature


def torsion(x, tck):
    derivs = []
    deriv2s = []
    deriv3s = []
    for coord in tck[1]:
        tck_coord = list(tck)
        tck_coord[1] = coord
        tck_coord = tuple(tck_coord)

        derivs.append(splev_deriv(x, tck_coord))
        deriv2s.append(splev_deriv2(x, tck_coord))
        deriv3s.append(splev_deriv3(x, tck_coord))

    deriv = np.stack(derivs, axis=1)
    deriv2 = np.stack(deriv2s, axis=1)
    deriv3 = np.stack(deriv3s, axis=1)

    cross = np.cross(deriv, deriv2)

    # Could be more efficient by only computing dot products of corresponding rows
    num = np.diag((cross @ deriv3.T))
    denom = np.linalg.norm(cross, axis=1) ** 2

    torsion = np.nan_to_num(num / denom)

    return torsion
