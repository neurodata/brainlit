import warnings
from tqdm import tqdm

import numpy as np
from skimage import morphology
import scipy.ndimage as ndi
from scipy.special import logsumexp
import random
from sklearn.neighbors import radius_neighbors_graph, KernelDensity
from brainlit.viz.swc2voxel import Bresenham3D
import networkx as nx
import math


class most_probable_neuron_path:
    def __init__(
        self,
        image,
        labels,
        soma_lbls=[],
        resolution=[0.3, 0.3, 1],
        coef_dist=0.5,
        coef_curv=0.0,
        frag_orientation_length=5,
    ):
        """Initialize object that performs tracing.

        Args:
            image (np.array): image
            labels (np.array): segmentation
            soma_lbls (list, optional): voxel coordinates of somas. Defaults to [].
            resolution (list, optional): voxel size. Defaults to [0.3, 0.3, 1].
            coef_dist (float, optional): hyperparameter that weights the distance factor. Defaults to 0.5.
            coef_curv (float, optional): hyperparameter that weights the curvature factor. Defaults to 0.0.
            frag_orientation_length (int, optional): length used to compute orientation at fragment endpoints. Defaults to 5.

        Raises:
            ValueError: Labels must have consecutive values.
        """

        # standard parameters
        self.image = image
        self.labels = labels
        self.frag_orientation_length = frag_orientation_length
        self.res = resolution
        self.coef_dist = coef_dist
        self.coef_curv = coef_curv

        # handling states and components
        num_components = np.amax(labels)
        self.num_components = num_components
        if np.amax(labels) != len(np.unique(labels)) - 1:
            raise ValueError("Labels input does not have consecutive values")
        num_states = num_components * 2 - len(soma_lbls)
        self.num_states = num_states
        self.soma_lbls = soma_lbls
        state_to_comp = {}  # dictionary of state to component
        # entry is tuple of "soma"/"fragment"
        # then component, then data
        comp_to_states = {}
        state_idx = 0
        for i in range(1, num_components + 1):
            if i in soma_lbls:
                state_to_comp[state_idx] = ("soma", i, None)
                comp_to_states[i] = [state_idx]
                state_idx += 1
            else:
                state_to_comp[state_idx] = (
                    "fragment",
                    i,
                    {
                        "coord1": None,
                        "orientation1": None,
                        "coord2": None,
                        "orientation2": None,
                        "soma connection point": None,
                    },
                )
                comp_to_states[i] = [state_idx, state_idx + 1]
                state_idx += 1
                state_to_comp[state_idx] = (
                    "fragment",
                    i,
                    {
                        "coord1": None,
                        "orientation1": None,
                        "coord2": None,
                        "orientation2": None,
                        "soma connection point": None,
                    },
                )
                state_idx += 1
        self.state_to_comp = state_to_comp
        self.comp_to_states = comp_to_states

        self.cost_mat_dist = np.ones((num_states, num_states)) * -np.inf
        self.cost_mat_int = np.ones((num_states, num_states)) * -np.inf
        self.cost_mat_cum = np.ones((num_states, num_states)) * -np.inf
        self.next_state_mat = -np.ones((num_states, num_states), dtype=int)
        self.dist_to_soma = np.ones(num_states) * np.inf

        self._compute_voxel_normals()

    def _compute_voxel_normals(self):
        """Compute the voxel normals for each soma and fragment."""
        normals = np.zeros((self.num_components + 1, 3))
        for lbl in range(1, self.num_components + 1):
            coords = np.array(np.where(self.labels == lbl)).T
            if len(coords) > 1:
                _, _, v = np.linalg.svd(coords - np.mean(coords, axis=0))
                normals[lbl] = v[0]
            else:
                normals[lbl] = np.array([0, 0, 0])
        self.voxel_normals = normals

    def point_point_dist(self, pt1, orientation1, pt2, orientation2, verbose=False):
        """Compute distance cost between two fragment objects.

        Args:
            pt1 (list): point on fragment 1
            orientation1 (list): orientation at pt1 on fragment 1
            pt2 (list): point on fragment 2
            orientation2 (list): orientation at pt2 on fragment 2
            verbose (bool, optional): Print the distance and its various components. Defaults to False.

        Raises:
            ValueError: If the points are the same, or the orientation vectors are not (roughly) unit length

        Returns:
            float: distance based cost
        """
        res = self.res

        dif = np.multiply(np.subtract(pt2, pt1), res)

        dist = np.linalg.norm(dif)

        if (
            dist == 0
            or not math.isclose(np.linalg.norm(orientation1), 1, abs_tol=1e-5)
            or not math.isclose(np.linalg.norm(orientation2), 1, abs_tol=1e-5)
        ):
            raise ValueError(
                f"pt1: {pt1} pt2: {pt2} dist: {dist}, o1: {orientation1} o2: {orientation2}"
            )

        if dist > 15:
            return np.inf

        k1_sq = 1 - np.dot(dif, orientation1) / dist
        k2_sq = 1 - np.dot(dif, orientation2) / dist

        k_cost = np.mean([k1_sq, k2_sq])

        if np.isnan(dist) or np.isnan(k_cost):
            raise ValueError(f"NAN cost: distance - {dist}, curv - {k_cost}")

        # if combined  average angle is tighter than 45 deg or either is tighter than 30 deg
        if 1 - k1_sq < -0.87 or 1 - k2_sq < -0.87:
            return np.inf

        cost = k_cost * self.coef_curv + self.coef_dist * (dist ** 2)
        if verbose:
            print(
                f"Distance: {dist}, Curv penalty: {k_cost} (dots {1-k1_sq}, {1-k2_sq}, from dif-{dif}), Total cost: {cost}"
            )

        return cost

    def point_blob_dist(self, point, orientation, blob_lbl, verbose=False):
        """Compute distance between a fragment object and a blob (soma) object

        Args:
            point (list): point on fragment
            orientation (list): orientation at point on fragment
            blob_lbl (int): label of blob (soma)

        Raises:
            ValueError: If orientation vector is not (roughly) unit length

        Returns:
            float: distance based cost
        """
        soma_coords = np.array(np.where(self.labels == blob_lbl)).T

        res = self.res

        diff = soma_coords - np.multiply(point, res)

        norms = np.linalg.norm(diff, axis=1)

        # find closest soma voxel
        idx = np.argmin(norms)

        soma_coord = soma_coords[idx]

        dist = norms[idx]

        if not math.isclose(np.linalg.norm(orientation), 1, abs_tol=1e-5):
            raise ValueError(f"pt: {point} dist: {dist}, o: {orientation}")

        k_sq = 1 - np.dot(diff[idx], orientation) / dist

        if np.isnan(dist) or np.isnan(k_sq):
            raise ValueError(f"NAN cost: distance - {dist}, curv - {k_sq}")

        # if the angle is more than 45 deg, discard
        if 1 - k_sq < -0.87:
            return np.inf

        cost = self.coef_curv * k_sq + self.coef_dist * (dist ** 2)

        if verbose:
            print(
                f"Distance: {dist}, Curv penalty: {k_sq} (dot {1-k_sq}, from diff-{diff[idx]}), Total cost: {cost}"
            )

        return cost

    def fragment_to_soma(
        self, fragment_state_idx, soma_state_idx, fragment_coord, soma_coord
    ):
        """Compute the distance between a fragment and a soma state.

        Args:
            fragment_state_idx (int): index of fragment state
            soma_state_idx (int): index of soma state
            fragment_coord (list): voxel coordinate of the fragment
            soma_coord (list): voxel coordinate of the soma

        Returns:
            float: distance based cost
        """
        fragment_orientation = self.state_to_comp[fragment_state_idx][2]["orientation1"]
        soma_orientation = self.state_to_comp[soma_state_idx][2]["orientation1"]

        return self.point_point_dist(
            fragment_coord,
            fragment_orientation,
            soma_coord,
            soma_orientation,
            verbose=False,
        )

    def connect_somas(self):
        """Connect somas to create a starting configuration.

        Returns:
            bool: True if successful
        """
        for soma_lbl in self.soma_lbls:
            soma_idx = self.comp_to_states[soma_lbl][0]
            self.dist_to_soma[soma_idx] = 0

        return True

    def connect_fragments(self):
        """Connect fragments to create a starting configuration.

        Returns:
            bool: True if successful
        """
        state_to_comp = self.state_to_comp
        comp_to_states = self.comp_to_states

        num_components = self.num_components
        voxel_normals = self.voxel_normals

        # first get all combinations of fragment pairs
        fragment_pairs = np.array(np.where(self.labels > 0)).T
        num_fragment_pairs = len(fragment_pairs)

        # loop over fragment pairs
        for idx in tqdm(range(num_fragment_pairs)):
            pair = fragment_pairs[idx]
            point1 = pair[0]
            point2 = pair[1]
            if point1 == point2:
                continue
            lbl1 = self.labels[point1[0], point1[1], point1[2]]
            lbl2 = self.labels[point2[0], point2[1], point2[2]]
            if lbl1 == lbl2:
                continue
            comp1 = lbl1
            comp2 = lbl2

            state_idx1, state_idx2 = comp_to_states[comp1][0], comp_to_states[comp2][0]

            if (
                state_to_comp[state_idx1][2]["coord1"] is None
                and state_to_comp[state_idx1][2]["coord2"] is None
            ):
                state_to_comp[state_idx1][2]["coord1"] = point1
                state_to_comp[state_idx1][2]["orientation1"] = voxel_normals[comp1]

            if (
                state_to_comp[state_idx2][2]["coord1"] is None
                and state_to_comp[state_idx2][2]["coord2"] is None
            ):
                state_to_comp[state_idx2][2]["coord1"] = point2
                state_to_comp[state_idx2][2]["orientation1"] = voxel_normals[comp2]

            if (
                state_to_comp[state_idx1][2]["coord1"] is not None
                and state_to_comp[state_idx2][2]["coord1"] is not None
            ):
                dist_cost = self.point_point_dist(
                    state_to_comp[state_idx1][2]["coord1"],
                    state_to_comp[state_idx1][2]["orientation1"],
                    state_to_comp[state_idx2][2]["coord1"],
                    state_to_comp[state_idx2][2]["orientation1"],
                )
                self.cost_mat_dist[state_idx1, state_idx2] = dist_cost
                self.cost_mat_dist[state_idx2, state_idx1] = dist_cost

        return True

    def connect_fragments_somas(self):
        """Connect fragments to somas to create a starting configuration.

        Returns:
            bool: True if successful
        """
        state_to_comp = self.state_to_comp
        comp_to_states = self.comp_to_states

        num_states = self.num_states
        num_components = self.num_components
        voxel_normals = self.voxel_normals

        # loop over fragments
        for i in tqdm(range(num_components + 1)):
            if i in self.soma_lbls:
                continue

            # loop over somas
            for j in self.soma_lbls:
                if i == j:
                    continue

                fragment_coord = np.array(np.where(self.labels == i)).T
                soma_coord = np.array(np.where(self.labels == j)).T
                fragment_orientation = voxel_normals[i]
                soma_orientation = voxel_normals[j]

                state_idx1, state_idx2 = comp_to_states[i][0], comp_to_states[j][0]

                if state_to_comp[state_idx1][2]["coord1"] is None:
                    state_to_comp[state_idx1][2]["coord1"] = fragment_coord[0]
                    state_to_comp[state_idx1][2]["orientation1"] = fragment_orientation

                if state_to_comp[state_idx2][2]["coord1"] is None:
                    state_to_comp[state_idx2][2]["coord1"] = soma_coord[0]
                    state_to_comp[state_idx2][2]["orientation1"] = soma_orientation

                if (
                    state_to_comp[state_idx1][2]["coord1"] is not None
                    and state_to_comp[state_idx2][2]["coord1"] is not None
                ):
                    dist_cost = self.point_point_dist(
                        state_to_comp[state_idx1][2]["coord1"],
                        state_to_comp[state_idx1][2]["orientation1"],
                        state_to_comp[state_idx2][2]["coord1"],
                        state_to_comp[state_idx2][2]["orientation1"],
                    )
                    self.cost_mat_dist[state_idx1, state_idx2] = dist_cost
                    self.cost_mat_dist[state_idx2, state_idx1] = dist_cost

        return True


if __name__ == "__main__":
    config = Config()
    config.load_from_file("config.yaml")
    config.print()

    fragments = Fragments(config)
    fragments.load_fragments("fragments.h5")
    fragments.print()

    fragments.build_graph()

    fragments.connect_somas()
    fragments.connect_fragments()
    fragments.connect_fragments_somas()
