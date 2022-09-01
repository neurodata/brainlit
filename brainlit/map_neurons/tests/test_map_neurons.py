from brainlit.map_neurons.map_neurons import *
from pathlib import Path
import os
import numpy as np
import h5py
from scipy.io import savemat
from scipy.interpolate import splprep, CubicHermiteSpline
import pytest
import pandas as pd


@pytest.fixture(scope="session")
def init_crt(tmp_path_factory) -> CloudReg_Transform:
    """Write the mat files for a simulated CloudReg registration computation.

    Returns:
        CloudReg_Transform: example CloudReg Transform that has an affine transform and whose nonlinear component is actually just a translation.
    """
    data_dir = tmp_path_factory.mktemp("data")

    # Save affine file
    path_matt_A = data_dir / "A.mat"

    # scale by 2 then add 10's
    A = np.eye(4)
    A[:3, :3] = np.eye(3) * 2
    A[:3, 3] = [10, 10, 10]
    A = np.linalg.inv(A)
    data = {"A": A}
    savemat(path_matt_A, data)

    # Save velocity file
    path_matt_v = data_dir / "v.mat"
    hf = h5py.File(path_matt_v, "w")

    nT = 10
    shp = [20, 20, 20]

    data = {}
    for comp in ["vtx", "vty", "vtz"]:
        ra = np.ones([nT] + shp)
        hf.create_dataset(comp, data=ra)

    crt = CloudReg_Transform(vpath=path_matt_v, Apath=path_matt_A)

    return crt


@pytest.fixture()
def init_gg() -> GeometricGraph:
    dict = {
        "x": [0, 1, 1],
        "y": [0, 0, 1],
        "z": [0, 0, 0],
        "sample": [1, 2, 3],
        "parent": [-1, 1, 2],
    }
    df = pd.DataFrame(data=dict)

    G = GeometricGraph(df=df)
    G.fit_spline_tree_invariant()

    return G


##############
### inputs ###
##############


def test_D_order(init_crt):
    crt = init_crt
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    derivs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])

    with pytest.raises(ValueError, match=r"Argument order must be 1, not 2"):
        transformed_derivs = crt.D(positions, derivs, order=2)


def test_compute_derivs_input():
    pts = np.random.uniform(size=(10, 3))
    tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=0)

    with pytest.raises(
        ValueError,
        match=r"When using spline method, positions should be None and tck should not",
    ):
        compute_derivs(us=u, tck=tck, positions=pts, deriv_method="spline")
    with pytest.raises(
        ValueError,
        match=r"When using difference method, tck should be None and positions should not",
    ):
        compute_derivs(us=u, tck=tck, positions=pts, deriv_method="difference")

    with pytest.raises(ValueError, match=r"Invalid deriv_method argument: foo"):
        compute_derivs(us=u, tck=tck, positions=pts, deriv_method="foo")


##################
### validation ###
##################


def test_applyaffine(init_crt):
    crt = init_crt
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    true_transformed_positions = np.array(
        [[10, 10, 10], [12, 10, 10], [10, 12, 10], [10, 10, 12], [12, 12, 12]]
    )

    transformed_positions = crt.apply_affine(positions)

    np.testing.assert_array_equal(true_transformed_positions, transformed_positions)


def test_evaluate(init_crt):
    crt = init_crt
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    true_transformed_positions = np.array(
        [[-1, -1, -1], [0, -1, -1], [-1, 0, -1], [-1, -1, 0], [0, 0, 0]]
    )

    transformed_positions = crt.evaluate(positions)

    np.testing.assert_almost_equal(true_transformed_positions, transformed_positions)


def test_D(init_crt):
    crt = init_crt
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    derivs = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]])
    true_transformed_derivs = derivs

    transformed_derivs = crt.D(positions, derivs)

    np.testing.assert_almost_equal(true_transformed_derivs, transformed_derivs)


def test_compute_derivs_spline():
    pts = np.random.uniform(size=(10, 3))
    tck, u = splprep([pts[:, 0], pts[:, 1], pts[:, 2]], s=0)
    true_derivs = np.array(splev(u, tck, der=1)).T

    derivs = compute_derivs(u, tck, deriv_method="spline")
    np.testing.assert_array_equal(true_derivs, derivs)


def test_compute_derivs_diff():
    us = np.array([0, 1, 2])
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    true_derivs = np.array([[1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 1, 0]])

    derivs = compute_derivs(us, positions=positions, deriv_method="difference")
    np.testing.assert_array_equal(true_derivs, derivs)

    us = np.array([0, 1, 3])
    positions = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    true_derivs = np.array(
        [[1, 0, 0], [4 / np.sqrt(17), 1 / np.sqrt(17), 0], [0, 1, 0]]
    )

    derivs = compute_derivs(us, positions=positions, deriv_method="difference")
    np.testing.assert_array_equal(true_derivs, derivs)


def test_transform_geometricgraph(init_crt, init_gg):
    G = init_gg
    ct = init_crt

    G = transform_geometricgraph(G, ct)
    true_positions = np.array([[-1, -1, -1], [0, -1, -1], [0, 0, -1]])
    true_derivs = np.array([[1, 0, 0], [1 / np.sqrt(2), 1 / np.sqrt(2), 0], [0, 1, 0]])

    positions = []
    derivs = []

    assert isinstance(G.spline_tree.nodes[0]["spline"][0], CubicHermiteSpline)

    for node in G.spline_tree.nodes[0]["path"]:
        positions.append(G.nodes[node]["loc"])
        derivs.append(G.nodes[node]["deriv"])

    np.testing.assert_almost_equal(true_positions, np.array(positions))
    np.testing.assert_almost_equal(true_derivs, np.array(derivs))
