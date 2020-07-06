from scipy.interpolate import splev
import numpy as np


def splev_deg0(x, xi, i):
    if i < len(xi) - 2:
        within = (x >= xi[i]) & (x < xi[i + 1])
    else:
        within = (x >= xi[i]) & (x <= xi[i + 1])

    return np.array(1 * (within))


def splev_degreecontrol(x, tck):
    if tck[2] < 0:
        return 0 * x
    elif tck[2] == 0:
        val = 0 * x
        cs = tck[1]
        xi = tck[0]
        for j, c in enumerate(cs):
            if c != 0:
                val = val + c * splev_deg0(x, xi, j)
        return val
    else:
        return splev(x, tck)


def splev_deriv(x, tck):
    cs = tck[1]
    p = tck[2]
    xi = tck[0]

    pre_xi = (xi[0] - 1) * np.ones((p + 1))
    post_xi = (xi[-1] + 1) * np.ones((p + 1))

    xi = np.concatenate((pre_xi, xi, post_xi))
    cs = np.concatenate((np.zeros(p + 1), cs, np.zeros(p + 1)))
    n = len(cs)
    deriv = np.zeros((len(x)))

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

        tckb1 = list(tck)
        cb1 = 0 * cs
        cb1[j] = 1
        tckb1[0] = xi
        tckb1[1] = cb1
        tckb1[2] = p - 1
        tckb1 = tuple(tckb1)

        tckb2 = list(tck)
        cb2 = 0 * cs
        cb2[j + 1] = 1
        tckb2[0] = xi
        tckb2[1] = cb2
        tckb2[2] = p - 1
        tckb2 = tuple(tckb2)

        deriv = deriv + cs[j] * (
            splev_degreecontrol(x, tckb1) * c1 - splev_degreecontrol(x, tckb2) * c2
        )
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
