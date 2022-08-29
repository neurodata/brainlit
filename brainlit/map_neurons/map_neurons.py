from logging import root
from brainlit.algorithms.trace_analysis.fit_spline import (
    GeometricGraph,
    compute_parameterization,
)
import typing
import numpy as np
import copy
import networkx as nx
from scipy.interpolate import (
    BSpline,
    splev,
    CubicHermiteSpline,
    RegularGridInterpolator,
)
import h5py
from tqdm import tqdm
import scipy.io as io


class DiffeomorphismAction:
    """Interface for differentiable mappings e.g. transformations that register a brain image to an atlas.
    """
    def evaluate(self, position: np.array) -> np.array:
        pass

    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        pass


class CloudReg_Transform(DiffeomorphismAction):
    """Object that can read mat files from CloudReg, and compute transformations on points and Jacobians.
    Implements DiffeomorphismAction which is an interface to transform points and tangent vectors.
    """

    def __init__(self, vpath: str, Apath: str):
        """Compute transformation from CloudReg mat files

        Args:
            vpath (str): path to mat file
        """
        # not: transformation files go from template space to target space
        f = h5py.File(vpath, "r")
        self.vtx = np.array(f.get("vtx"))
        self.vty = np.array(f.get("vty"))
        self.vtz = np.array(f.get("vtz"))

        f = io.loadmat(Apath)
        A = f["A"]
        self.A = A
        self.B = np.linalg.inv(A)

        self._integrate()

    def apply_affine(self, position: np.array) -> np.array:
        """Apply affine transformation in the transformation of positions in target space to atlas space.

        Args:
            position (np.array): nx3 array with positions in target space.

        Returns:
            np.array: positions after affine transformation was applied.
        """
        # transformation direction: atlas
        A = self.B
        transformed_position = (
            A[:3, :3] @ position.T + np.expand_dims(A[:3, 3], axis=0).T
        )
        transformed_position = transformed_position.T

        return transformed_position

    def _integrate(self, velocity_voxel_size: list=[100.0, 100.0, 100.0]):
        """Integrate velocity field in order to compute diffeomorphsm mapping. Translated from https://github.com/neurodata/CloudReg/blob/master/cloudreg/registration/transform_points.m
        Integration is done in the direction to allow mapping from target to atlas space.

        Args:
            velocity_voxel_size (list, optional): Voxel resolution of trarnsformation. Defaults to [100.0, 100.0, 100.0].
        """

        vtx = self.vtx
        vty = self.vty
        vtz = self.vtz

        nT = vtx.shape[0]
        dt = 1 / nT

        nxV = vtx.shape[1:]
        dxV = velocity_voxel_size
        xV = np.arange(0, nxV[0]) * dxV[0]
        yV = np.arange(0, nxV[1]) * dxV[1]
        zV = np.arange(0, nxV[2]) * dxV[2]
        xV = xV - np.mean(xV)
        yV = yV - np.mean(yV)
        zV = zV - np.mean(zV)

        [XV, YV, ZV] = np.meshgrid(xV, yV, zV, indexing="ij")
        timesteps = np.arange(0, nT, 1)
        indicator = -1

        transx = XV
        transy = YV
        transz = ZV

        for t in tqdm(timesteps, desc="integrating velocity field"):
            Xs = XV + indicator * vtx[t, :, :, :] * dt
            Ys = YV + indicator * vty[t, :, :, :] * dt
            Zs = ZV + indicator * vtz[t, :, :, :] * dt
            F = RegularGridInterpolator(
                (xV, yV, zV),
                transx - XV,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            XYZs = np.reshape(np.stack((Xs, Ys, Zs), axis=-1), newshape=(-1, 3))
            transx = np.reshape(F(XYZs), Xs.shape) + Xs
            F = RegularGridInterpolator(
                (xV, yV, zV),
                transy - YV,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            transy = np.reshape(F(XYZs), Ys.shape) + Ys
            F = RegularGridInterpolator(
                (xV, yV, zV),
                transz - ZV,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            transz = np.reshape(F(XYZs), Zs.shape) + Zs

        Fx = RegularGridInterpolator(
            (xV, yV, zV),
            transx - XV,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        Fy = RegularGridInterpolator(
            (xV, yV, zV),
            transy - YV,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        Fz = RegularGridInterpolator(
            (xV, yV, zV),
            transz - ZV,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        self.og_coords = [XV, YV, ZV]
        self.diffeomorphism = (Fx, Fy, Fz)

    def evaluate(self, position: np.array) -> np.array:
        """Apply non-affine component of mapping to positions.

        Args:
            position (np.array): Positions at which to compute mappings.

        Returns:
            np.array: Mappings of the input.
        """
        Fx = self.diffeomorphism[0]
        Fy = self.diffeomorphism[1]
        Fz = self.diffeomorphism[2]

        transformed_positionx = Fx(position) + position[:, 0]
        transformed_positiony = Fy(position) + position[:, 1]
        transformed_positionz = Fz(position) + position[:, 2]

        transformed_position = np.stack(
            (transformed_positionx, transformed_positiony, transformed_positionz),
            axis=1,
        )

        return transformed_position

    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        """Compute transformed derivatives of mapping at given positions. Only for the non-affine component.

        Args:
            position (np.array): nx3 positions at which to compute derivatives.
            deriv (np.array): nx3 derivatives at the respective positions.
            order (int, optional): Order of derivative (must be 1). Defaults to 1.

        Raises:
            ValueError: Only derivative order 1 is allowed here.

        Returns:
            np.array: Transformed derivatives
        """

        if order != 1:
            raise ValueError(f"Argument order must be 1, not {1}")

        Fx = self.diffeomorphism[0]
        Fy = self.diffeomorphism[1]
        Fz = self.diffeomorphism[2]

        transformed_deriv = deriv.copy()
        step = 1
        for i, (pos, d) in enumerate(zip(position, deriv)):
            J = np.zeros((3, 3))
            J[0, 0] = (Fx([pos[0] + step, pos[1], pos[2]]) + step - Fx(pos)) / step
            J[0, 1] = (Fx([pos[0], pos[1] + step, pos[2]]) - Fx(pos)) / step
            J[0, 2] = (Fx([pos[0], pos[1], pos[2] + step]) - Fx(pos)) / step

            J[1, 0] = (Fy([pos[0] + step, pos[1], pos[2]]) - Fy(pos)) / step
            J[1, 1] = (Fy([pos[0], pos[1] + step, pos[2]]) + step - Fy(pos)) / step
            J[1, 2] = (Fy([pos[0], pos[1], pos[2] + step]) - Fy(pos)) / step

            J[2, 0] = (Fz([pos[0] + step, pos[1], pos[2]]) - Fz(pos)) / step
            J[2, 1] = (Fz([pos[0], pos[1] + step, pos[2]]) - Fz(pos)) / step
            J[2, 2] = (Fz([pos[0], pos[1], pos[2] + step]) + step - Fz(pos)) / step
            transformed_deriv[i, :] = np.matmul(J, d).T

        return transformed_deriv

def compute_derivs(us: np.array, tck: tuple, positions: np.array, deriv_method: str="difference") -> np.array:
    """Estimate derivatives of a spline parameterized by scipy's spline API.

    Args:
        us (np.array): Parameter values (in form returned by scipy.interpolate.splprep).
        tck (tuple): Knots, bspline coefficients, and degree of spline (in form returned by scipy.interpolate.splprep).
        positions (np.array): nx3 array of positions (for use by difference method).
        deriv_method (str, optional): Method to use, spline for scipy.interpolate.splev or difference for  . Defaults to "difference".

    Raises:
        ValueError: If derivative method is unrecognized.

    Returns:
        np.array: Derivative estimates at specified positions.
    """
    if deriv_method == "spline":
        derivs = np.array(splev(us, tck, der=1)).T
    elif deriv_method == "difference":
        # Sundqvist & Veronis 1970
        f_im1 = positions[:-2, :]
        f_i = positions[1:-1, :]
        f_ip1 = positions[2:, :]
        hs = np.diff(us)
        h_im1 = np.expand_dims(hs[:-1], axis=1)
        h_i = np.expand_dims(hs[1:], axis=1)

        if len(us) >= 3:
            diffs = (
                f_ip1
                - np.multiply((1 - np.divide(h_i, h_im1) ** 2), f_i)
                - np.multiply(np.divide(h_i, h_im1) ** 2, f_im1)
            )
            diffs = np.concatenate(
                (
                    [positions[1, :] - positions[0, :]],
                    diffs,
                    [positions[-1, :] - positions[-2, :]],
                ),
                axis=0,
            )
        elif len(us) == 2:
            diffs = np.array(
                [positions[1, :] - positions[0, :], positions[-1, :] - positions[-2, :]]
            )
        norms = np.linalg.norm(diffs, axis=1)
        derivs = np.divide(diffs, np.array([norms]).T)
    else:
        raise ValueError(f"Invalid deriv_method argument: {deriv_method}")

    return derivs


def transform_GeometricGraph(
    G_transformed: GeometricGraph,
    Phi: DiffeomorphismAction,
    deriv_method: str="difference",
    derivs: np.arrry=None,
) -> GeometricGraph:
    """Apply a diffeomorphism to a GeometricGraph object.

    Args:
        G_transformed (GeometricGraph): Object that will be transformed.
        Phi (DiffeomorphismAction): Diffeomorphism that will define the transformation.
        deriv_method (str, optional): Derivative method to use in compute_derivs if derivs argument is None. Defaults to "difference".
        derivs (np.array, optional): nx3 array of derivative values associated with nodes on the GemoetricGraph. Only applicable if GeometricGraph has a single branch. Defaults to None.

    Raises:

    Raises:
        NotImplementedError: This method transforms GemoetricGraphs composed of BSplines to GemoetricGraphs composed of CubicHermite splines.
        ValueError: If derivs argument is given but GemoetricGraph has multiple branches.
        ValueError: If splines were not computed for the GemoetricGraph yet.

    Returns:
        GeometricGraph: Transformed GeometricGraph
    """
    if G_transformed.spline_type is not BSpline:
        raise NotImplementedError("Can only transform bsplines")

    if derivs is not None and len(G_transformed.spline_tree.nodes) > 1:
        raise ValueError("Manually gave derivatives for tree with multiple branches")

    if not G_transformed.spline_tree:
        raise ValueError(
            "Must compute spline tree before running this function - fit_spline_tree_invariant()"
        )
    # if "deriv" not in G_transformed.nodes[G_transformed.root].keys():
    #     raise ValueError("Must compute derivatives before running this function - compute_derivs(deriv_method)")

    spline_tree = G_transformed.spline_tree

    # process in reverse dfs order to ensure parents are processed after
    reverse_dfs = list(reversed(list(nx.topological_sort(spline_tree))))

    for node_n, node in enumerate(reverse_dfs):
        path = spline_tree.nodes[node]["path"]
        tck, us = spline_tree.nodes[node]["spline"]
        positions = np.array(splev(us, tck, der=0)).T
        if derivs is None or node_n > 0:
            derivs = compute_derivs(us, tck, positions, deriv_method=deriv_method)

        transformed_positions = Phi.evaluate(positions)
        transformed_us = compute_parameterization(transformed_positions)
        transformed_derivs = Phi.D(positions, derivs, order=1)
        norms = np.linalg.norm(transformed_derivs, axis=1)
        transformed_derivs = np.divide(transformed_derivs, np.array([norms]).T)

        chspline = CubicHermiteSpline(
            transformed_us, transformed_positions, transformed_derivs
        )

        spline_tree.nodes[node]["spline"] = chspline, transformed_us

        for trace_node, position, deriv in zip(
            path, transformed_positions, transformed_derivs
        ):
            G_transformed.nodes[trace_node]["loc"] = position
            G_transformed.nodes[trace_node]["deriv"] = deriv

    return G_transformed
