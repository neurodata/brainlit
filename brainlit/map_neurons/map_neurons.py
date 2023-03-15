from logging import root
from brainlit.algorithms.trace_analysis.fit_spline import (
    GeometricGraph,
    compute_parameterization,
    CubicHermiteChain,
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
    """Interface for differentiable mappings e.g. transformations that register a brain image to an atlas."""

    def evaluate(self, position: np.array) -> np.array:
        """Evaluate the mapping at specified positions.

        Args:
            position (np.array): Coordinates in original space.

        Returns:
            np.array: Transformed coordinates.
        """
        pass

    def D(self, position: np.array, deriv: np.array, order: int = 1) -> np.array:
        """Evaluate the mapping on a set of derivatives at specified positions.

        Args:
            position (np.array): Coordinates in the original space.
            deriv (np.array): Derivatives at the respective positions
            order (int, optional): Derivative order. Defaults to 1.

        Returns:
            np.array: Transformed derivatives.
        """
        pass


class CloudReg_Transform(DiffeomorphismAction):
    """Object that can read mat files from CloudReg, and compute transformations on points and Jacobians.
    Implements DiffeomorphismAction which is an interface to transform points and tangent vectors.
    """

    def __init__(self, vpath: str, Apath: str, direction: str = "atlas"):
        """Compute transformation from CloudReg mat files

        Args:
            vpath (str): path to mat file.
            Apath (str): path to mat file with affine transformation.
            direction (str, optional): direction of transformation, only target to atlas space is implemented so far. Defaults to "atlas".
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

        self.direction = direction

        self._integrate(direction=direction)

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

    def _integrate(
        self,
        velocity_voxel_size: list = [100.0, 100.0, 100.0],
        direction: str = "atlas",
    ):
        """Integrate velocity field in order to compute diffeomorphsm mapping. Translated from https://github.com/neurodata/CloudReg/blob/master/cloudreg/registration/transform_points.m
        Integration is done in the direction to allow mapping from target to atlas space.

        Args:
            velocity_voxel_size (list, optional): Voxel resolution of trarnsformation. Defaults to [100.0, 100.0, 100.0].
            direction (str, optional): direction of transformation, only target to atlas space is implemented so far. Defaults to "atlas".

        Raises:
            NotImplementedError: direction must be to atlas space.
            ValueError: invalid value for direction
        """

        vtx = self.vtx
        vty = self.vty
        vtz = self.vtz

        nT = vtx.shape[0]
        dt = 1 / nT

        nxV = [vtx.shape[2], vtx.shape[3], vtx.shape[1]]

        dxV = velocity_voxel_size  # units: microns/voxel
        xV = np.arange(0, nxV[0]) * dxV[0]  # units: microns
        yV = np.arange(0, nxV[1]) * dxV[1]
        zV = np.arange(0, nxV[2]) * dxV[2]
        xV = xV - np.mean(xV)
        yV = yV - np.mean(yV)
        zV = zV - np.mean(zV)
        # *V variables represent a static grid
        [XV, YV, ZV] = np.meshgrid(xV, yV, zV, indexing="ij")  # units: microns
        # reshape to match matlab
        XV = np.swapaxes(XV, 0, 1)
        YV = np.swapaxes(YV, 0, 1)
        ZV = np.swapaxes(ZV, 0, 1)

        if direction == "atlas":
            timesteps = np.arange(0, nT, 1)
            indicator = -1
        elif direction == "target":
            timesteps = np.arange(nT - 1, -1, -1)
            indicator = 1
            raise NotImplementedError(
                f"Cannot integrate from atlas to target space yet."
            )
        else:
            raise ValueError(
                f"direction argument must be atlas or target, not {direction}"
            )

        # trans variables aggregates the cumulative displacement from the originial grid coordinates
        transx = XV  # units: microns
        transy = YV
        transz = ZV

        vtx = np.swapaxes(vtx, 0, 3)
        vtx = np.swapaxes(vtx, 1, 2)
        vty = np.swapaxes(vty, 0, 3)
        vty = np.swapaxes(vty, 1, 2)
        vtz = np.swapaxes(vtz, 0, 3)
        vtz = np.swapaxes(vtz, 1, 2)

        for t in tqdm(timesteps, desc="integrating velocity field", disable=False):
            # Deform the static grid at a certain time point
            Xs = XV + indicator * vtx[:, :, :, t] * dt
            Ys = YV + indicator * vty[:, :, :, t] * dt
            Zs = ZV + indicator * vtz[:, :, :, t] * dt
            F = RegularGridInterpolator(
                (yV, xV, zV),
                transx - XV,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )

            XYZs = np.reshape(np.stack((Ys, Xs, Zs), axis=-1), newshape=(-1, 3))
            # Add the newest timestep of displacement (Xs) to the cumulative displacement
            transx = np.reshape(F(XYZs), Xs.shape) + Xs
            F = RegularGridInterpolator(
                (yV, xV, zV),
                transy - YV,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            transy = np.reshape(F(XYZs), Ys.shape) + Ys
            F = RegularGridInterpolator(
                (yV, xV, zV),
                transz - ZV,
                method="linear",
                bounds_error=False,
                fill_value=None,
            )
            transz = np.reshape(F(XYZs), Zs.shape) + Zs

        Fx = RegularGridInterpolator(
            (yV, xV, zV),
            transx - XV,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        Fy = RegularGridInterpolator(
            (yV, xV, zV),
            transy - YV,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )
        Fz = RegularGridInterpolator(
            (yV, xV, zV),
            transz - ZV,
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        self.og_coords = [ZV, XV, YV]
        self.diffeomorphism = (Fx, Fy, Fz)

    def evaluate(self, position: np.array) -> np.array:
        """Apply non-affine component of mapping to positions, in direction of self.direction (default from target to atlas).

        Args:
            position (np.array): Positions at which to compute mappings.

        Returns:
            np.array: Mappings of the input.
        """
        Fx = self.diffeomorphism[0]
        Fy = self.diffeomorphism[1]
        Fz = self.diffeomorphism[2]

        transformed_positionx = Fx(position[:, [1, 0, 2]]) + position[:, 0]
        transformed_positiony = Fy(position[:, [1, 0, 2]]) + position[:, 1]
        transformed_positionz = Fz(position[:, [1, 0, 2]]) + position[:, 2]

        transformed_position = np.stack(
            (transformed_positionx, transformed_positiony, transformed_positionz),
            axis=1,
        )

        return transformed_position

    def Jacobian(self, pos: np.array) -> np.array:
        """Compute Jacobian of transformation at a given point.

        Args:
            pos (np.array): Coordinate in space.

        Returns:
            np.array: Jacobian at that coordinate
        """
        step = 1
        Fx = self.diffeomorphism[0]
        Fy = self.diffeomorphism[1]
        Fz = self.diffeomorphism[2]

        J = np.zeros((3, 3))
        J[0, 0] = (Fx([pos[1], pos[0] + step, pos[2]]) - Fx(pos[[1, 0, 2]])) / step + 1
        J[0, 1] = (Fx([pos[1] + step, pos[0], pos[2]]) - Fx(pos[[1, 0, 2]])) / step
        J[0, 2] = (Fx([pos[1], pos[0], pos[2] + step]) - Fx(pos[[1, 0, 2]])) / step

        J[1, 0] = (Fy([pos[1], pos[0] + step, pos[2]]) - Fy(pos[[1, 0, 2]])) / step
        J[1, 1] = (Fy([pos[1] + step, pos[0], pos[2]]) - Fy(pos[[1, 0, 2]])) / step + 1
        J[1, 2] = (Fy([pos[1], pos[0], pos[2] + step]) - Fy(pos[[1, 0, 2]])) / step

        J[2, 0] = (Fz([pos[1], pos[0] + step, pos[2]]) - Fz(pos[[1, 0, 2]])) / step
        J[2, 1] = (Fz([pos[1] + step, pos[0], pos[2]]) - Fz(pos[[1, 0, 2]])) / step
        J[2, 2] = (Fz([pos[1], pos[0], pos[2] + step]) - Fz(pos[[1, 0, 2]])) / step + 1

        return J

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
            raise ValueError(f"Argument order must be 1, not {order}")

        transformed_deriv = deriv.copy()
        for i, (pos, d) in enumerate(zip(position, deriv)):
            J = self.Jacobian(pos)
            transformed_deriv[i, :] = np.matmul(J, d).T

        return transformed_deriv


def compute_derivs(
    us: np.array,
    tck: tuple = None,
    positions: np.array = None,
    deriv_method: str = "difference",
) -> np.array:
    """Estimate derivatives of a sequence of points. Derivatives can be estimated in three ways:
    - For curves parameterized by scipy's spline API, spline estimation uses scipy's derivative computation
    - For a sequence of points, we use the finite-difference method from:

    Sundqvist, H., & Veronis, G. (1970). A simple finite‐difference grid with non‐constant intervals. Tellus, 22(1), 26-31.

    - one-sided derivatives are derived from the piecewise linear interpolation.

    Args:
        us (np.array): Parameter values (in form returned by scipy.interpolate.splprep).
        tck (tuple): Knots, bspline coefficients, and degree of spline (in form returned by scipy.interpolate.splprep).
        positions (np.array): nx3 array of positions (for use by difference method).
        deriv_method (str, optional): Method to use (from list above), spline for scipy.interpolate.splev, difference for the finite difference method, two-sided for one-sided derivatives from linear interpolation. Defaults to "difference".

    Raises:
        ValueError: If the wrong combination of arguments/deriv_method is used.
        ValueError: If derivative method is unrecognized.

    Returns:
        np.array: Derivative estimates at specified positions, or tuple of np.array for two-sided option.
    """
    if deriv_method == "spline":
        if tck is None or positions is not None:
            raise ValueError(
                f"When using spline method, positions should be None and tck should not"
            )
        derivs = np.array(splev(us, tck, der=1)).T
        return derivs
    elif deriv_method == "difference":
        if tck is not None or positions is None:
            raise ValueError(
                f"When using difference method, tck should be None and positions should not"
            )
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
        return derivs
    elif deriv_method == "two-sided":
        f_im1 = positions[:-1, :]
        f_i = positions[1:, :]
        left_derivs = f_i - f_im1
        norms = np.linalg.norm(left_derivs, axis=1)
        left_derivs = np.divide(left_derivs, np.array([norms]).T)
        right_derivs = f_i - f_im1
        norms = np.linalg.norm(right_derivs, axis=1)
        right_derivs = np.divide(right_derivs, np.array([norms]).T)
        return left_derivs, right_derivs
    else:
        raise ValueError(f"Invalid deriv_method argument: {deriv_method}")


def transform_geometricgraph(
    G_transformed: GeometricGraph,
    Phi: DiffeomorphismAction,
    deriv_method: str = "two-sided",
    derivs: np.array = None,
) -> GeometricGraph:
    """Apply a diffeomorphism to a GeometricGraph object.

    Args:
        G_transformed (GeometricGraph): Object that will be transformed.
        Phi (DiffeomorphismAction): Diffeomorphism that will define the transformation.
        deriv_method (str, optional): Choice of derivative estimation to use. Defaults to None.
        derivs (np.array, optional): nx3 array of derivative values associated with nodes on the GemoetricGraph. Only applicable if GeometricGraph has a single branch. Defaults to None.

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

    spline_tree = G_transformed.spline_tree

    # process in reverse dfs order to ensure parents are processed after
    reverse_dfs = list(reversed(list(nx.topological_sort(spline_tree))))

    if deriv_method in ["spline", "difference"]:
        for node_n, node in enumerate(reverse_dfs):
            path = spline_tree.nodes[node]["path"]
            tck, us = spline_tree.nodes[node]["spline"]
            positions = np.array(splev(us, tck, der=0)).T
            if derivs is None or node_n > 0:
                derivs = compute_derivs(
                    us=us, positions=positions, deriv_method=deriv_method
                )

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
    elif deriv_method == "two-sided":
        for node_n, node in enumerate(reverse_dfs):
            path = spline_tree.nodes[node]["path"]
            tck, us = spline_tree.nodes[node]["spline"]
            positions = np.array(splev(us, tck, der=0)).T

            # here, left derivs are the derivatives on the left side of the segment (i.e. the right derivative of the point on the left)
            if derivs is None:
                left_derivs, right_derivs = compute_derivs(
                    us=us, positions=positions, deriv_method=deriv_method
                )
            else:
                left_derivs = derivs[0]
                right_derivs = derivs[1]

            transformed_positions = Phi.evaluate(positions)
            transformed_us = compute_parameterization(transformed_positions)

            transformed_left_derivs = Phi.D(positions[:-1, :], left_derivs, order=1)
            norms = np.linalg.norm(transformed_left_derivs, axis=1)
            transformed_left_derivs = np.divide(
                transformed_left_derivs, np.array([norms]).T
            )

            transformed_right_derivs = Phi.D(positions[1:, :], right_derivs, order=1)
            norms = np.linalg.norm(transformed_right_derivs, axis=1)
            transformed_right_derivs = np.divide(
                transformed_right_derivs, np.array([norms]).T
            )

            chspline = CubicHermiteChain(
                transformed_us,
                transformed_positions,
                transformed_left_derivs,
                transformed_right_derivs,
            )

            spline_tree.nodes[node]["spline"] = chspline, transformed_us

            for i, (trace_node, position) in enumerate(
                zip(path, transformed_positions)
            ):
                G_transformed.nodes[trace_node]["loc"] = position
                # here, left deriv is the left side derivative
                if i > 0:
                    G_transformed.nodes[trace_node][
                        "left_deriv"
                    ] = transformed_right_derivs[i - 1, :]
                if i < len(path) - 1:
                    G_transformed.nodes[trace_node][
                        "right_deriv"
                    ] = transformed_left_derivs[i, :]

    return G_transformed
