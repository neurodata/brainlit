import numpy as np
import math
import zarr
from joblib import Parallel, delayed
from tqdm import tqdm
from brainlit.viz.swc2voxel import Bresenham3D
from brainlit.preprocessing import image_process
import networkx as nx
from typing import List, Tuple, Callable
from pathlib import Path
import pickle
import os
import copy
import itertools
from scipy.spatial import cKDTree


def _curv_dist(
    res: List[float],
    pt1: List[int],
    orientation1: List[int],
    pt2: List[int],
    orientation2: List[int],
):
    """Compute components of transition cost between two fragment states

    Args:
        res (list of floats): resolution of image
        pt1 (list of ints): first coordinate
        orientation1 (list of ints): orientation at first coordinate
        pt2 (list of ints): second coordinate
        orientation2 (list of ints): orientation at second coordinate

    Raises:
        ValueError: if an orientation is not unit length
        ValueError: if distance or curvature cost is nan

    Returns:
        [float]: cost of transition
    """
    dif = np.multiply(np.subtract(pt2, pt1), res)

    dist = np.linalg.norm(dif)

    if dist > 15:
        return np.inf, np.inf

    if (
        dist == 0
        or not math.isclose(np.linalg.norm(orientation1), 1, abs_tol=1e-5)
        or not math.isclose(np.linalg.norm(orientation2), 1, abs_tol=1e-5)
    ):
        raise ValueError(
            f"pt1: {pt1} pt2: {pt2} dist: {dist}, o1: {orientation1} o2: {orientation2}"
        )

    k1_sq = 1 - np.dot(dif, orientation1) / dist
    k2_sq = 1 - np.dot(dif, orientation2) / dist

    k_cost = np.mean([k1_sq, k2_sq])

    if np.isnan(dist) or np.isnan(k_cost):
        raise ValueError(f"NAN cost: distance - {dist}, curv - {k_cost}")

    # if combined  average angle is tighter than 45 deg or either is tighter than 30 deg
    if 1 - k1_sq < -0.87 or 1 - k2_sq < -0.87:
        return np.inf, np.inf
    else:
        return dist, k_cost


def _compute_dist_cost(pair, res, coef_dist=10, coef_curv=1000):
    state1_data = pair[0]
    state1 = state1_data[0]
    state1_dict = state1_data[1]

    state2_data = pair[1]
    state2 = state2_data[0]
    state2_dict = state2_data[1]

    if state1_dict["fragment"] == state2_dict["fragment"]:
        return (state1, state2, np.inf)
    elif state1_dict["type"] == "fragment" and state2_dict["type"] == "fragment":
        pt1 = state1_dict["point2"]
        orientation1 = state1_dict["orientation2"]
        pt2 = state2_dict["point1"]
        orientation2 = state2_dict["orientation1"]

        dist, k_cost = _curv_dist(
            res=res,
            pt1=pt1,
            orientation1=orientation1,
            pt2=pt2,
            orientation2=orientation2,
        )
        cost = coef_dist * (dist**2) + coef_curv * k_cost
        return (state1, state2, cost)
    else:
        raise ValueError("no two fragments?")


def _line_int_coord(loc1: List[int], loc2: List[int], tiered_path: str):
    image_tiered = zarr.open(tiered_path, mode="r")
    corner1 = [np.amin([loc1[i], loc2[i]]) for i in range(len(loc1))]
    corner2 = [np.amax([loc1[i], loc2[i]]) for i in range(len(loc1))]

    image_tiered_cutout = image_tiered[
        corner1[0] : corner2[0] + 1,
        corner1[1] : corner2[1] + 1,
        corner1[2] : corner2[2] + 1,
    ]

    loc1 = [int(loc1[i]) - corner1[i] for i in range(len(loc1))]
    loc2 = [int(loc2[i]) - corner1[i] for i in range(len(loc1))]

    xlist, ylist, zlist = Bresenham3D(
        int(loc1[0]),
        int(loc1[1]),
        int(loc1[2]),
        int(loc2[0]),
        int(loc2[1]),
        int(loc2[2]),
    )
    # exclude first and last points because they are included in the component intensity sum
    xlist = xlist[1:-1]
    ylist = ylist[1:-1]
    zlist = zlist[1:-1]

    sum = np.sum(image_tiered_cutout[xlist, ylist, zlist])

    return sum


def _compute_int_cost(pair, tiered_path):
    state1_data = pair[0]
    state1 = state1_data[0]
    state1_dict = state1_data[1]

    state2_data = pair[1]
    state2 = state2_data[0]
    state2_dict = state2_data[1]

    if state1_dict["fragment"] == state2_dict["fragment"]:
        return (state1, state2, np.inf)
    elif state1_dict["type"] == "fragment" and state2_dict["type"] == "fragment":
        int_cost = _line_int_coord(
            state1_dict["point2"], state2_dict["point1"], tiered_path=tiered_path
        )
        return (state1, state2, int_cost)

    else:
        raise ValueError("No two fragments?")


class ViterBrain:
    def __init__(
        self,
        G: nx.Graph,
        tiered_path: str,
        fragment_path: str,
        resolution: List[float],
        coef_curv: float,
        coef_dist: float,
        coef_int: float,
        parallel: int = 1,
    ) -> None:
        """Initialize ViterBrain object

        Args:
            G (nx.Graph): networkx graph representation of states
            tiered_path (str): path to tiered image
            fragment_path (str): path to fragments image
            resolution (list of floats): resolution of images
            coef_curv (float): curvature coefficient
            coef_dist (float): distance coefficient
            coef_int (float): image likelihood coefficient
            parallel (int, optional): Number of threads to use for parallelization. Defaults to 1.
        """
        self.nxGraph = G
        self.num_states = G.number_of_nodes()
        self.tiered_path = tiered_path
        self.fragment_path = fragment_path
        self.resolution = resolution
        self.coef_curv = coef_curv
        self.coef_dist = coef_dist
        self.coef_int = coef_int
        self.parallel = parallel

        soma_fragment2coords = {}
        for node in G.nodes:
            if G.nodes[node]["type"] == "soma":
                soma_fragment2coords[G.nodes[node]["fragment"]] = G.nodes[node][
                    "soma_coords"
                ]

        self.soma_fragment2coords = soma_fragment2coords

        comp_to_states = {}
        for node in G.nodes:
            frag = G.nodes[node]["fragment"]
            if frag in comp_to_states.keys():
                prev = comp_to_states[frag]
                states = prev + [node]
                comp_to_states[frag] = states
            else:
                comp_to_states[frag] = [node]
        self.comp_to_states = comp_to_states

    def frag_frag_dist_simple(
        self,
        state1: int,
        state2: int,
        verbose: bool = False,
    ) -> float:
        G = self.nxGraph
        res = self.resolution

        pt1 = G.nodes[state1]["point1"]
        pt2 = G.nodes[state1]["point2"]
        pt3 = G.nodes[state2]["point1"]
        dif = np.multiply(np.subtract(pt3, pt2), res)
        dist2 = np.linalg.norm(dif)
        if dist2 > 20:
            return np.inf

        dif = np.multiply(np.subtract(pt2, pt1), res)
        dist1 = np.linalg.norm(dif)

        return dist1 + dist2**2

    def frag_soma_dist(
        self,
        point: List[float],
        orientation: List[float],
        soma_lbl: int,
        verbose: bool = False,
    ) -> Tuple[float, List]:
        """Compute cost of transition from fragment state to soma state

        Args:
            point (list of floats): coordinate on fragment
            orientation (list of floats): orientation at fragment
            soma_lbl (int): label of soma component
            verbose (bool, optional): Prints cost values. Defaults to False.

        Raises:
            ValueError: if either distance or curvature cost is nan
            ValueError: if the computed closest soma coordinate is not on the soma

        Returns:
            [float]: cost of transition
            [list of floats]: closest soma coordinate
        """
        coords = self.soma_fragment2coords[soma_lbl]
        image_fragment = zarr.open_array(self.fragment_path, mode="r")

        difs = np.multiply(np.subtract(coords, point), self.resolution)
        dists = np.linalg.norm(difs, axis=1)
        argmin = np.argmin(dists)
        dif = difs[argmin, :]
        dist = dists[argmin]

        dot = np.dot(dif, orientation) / (
            np.linalg.norm(dif) * np.linalg.norm(orientation)
        )
        k_cost = 1 - dot

        if np.isnan(k_cost) or np.isnan(dist):
            raise ValueError(f"NAN cost: distance - {dist}, curv - {k_cost}")

        if dist > 15:
            cost = np.inf
        else:
            cost = k_cost * self.coef_curv + self.coef_dist * (dist**2)

        nonline_point = coords[argmin, :]
        if (
            image_fragment[
                nonline_point[0],
                nonline_point[1],
                nonline_point[2],
            ]
            != soma_lbl
        ):
            raise ValueError("Soma point is not on soma")
        if verbose:
            print(
                f"Distance: {dist}, Curv penalty: {k_cost}, Total cost: {cost}, connection point: {nonline_point}"
            )

        return cost, nonline_point

    def compute_all_costs_dist(self, data_dir: str) -> None:
        """Splits up transition computation tasks then assembles them into networkx graph

        Args:
            frag_frag_func (function): function that computes transition cost between fragments
            frag_soma_func (function): function that computes transition cost between fragments
        """
        parallel = self.parallel
        G = self.nxGraph

        isExist = os.path.exists(data_dir)
        if not isExist:
            print(f"Creating directory: {data_dir}")
            os.makedirs(data_dir)
        else:
            print(f"Data will be stored in {data_dir}")

        data = []
        for state in range(self.num_states):
            data.append(np.multiply(G.nodes[state]["point2"], self.resolution))
        data = np.stack(data, axis=0)
        kdt1 = cKDTree(data)
        data = []
        for state in range(self.num_states):
            data.append(np.multiply(G.nodes[state]["point1"], self.resolution))
        data = np.stack(data, axis=0)
        kdt2 = cKDTree(data)
        results = kdt1.query_ball_tree(kdt2, r=15)

        pairs = []
        for state1, nbrs in enumerate(tqdm(results, desc="constructing pairs")):
            state1_data = (state1, G.nodes[state1])
            for state2 in nbrs:
                state2_data = (state2, G.nodes[state2])
                pairs.append((state1_data, state2_data))

        print(f"{len(pairs)} for {self.num_states} states")

        chunk_size = 100000
        for start in tqdm(range(0, len(pairs), chunk_size), desc="pair chunks"):
            pairs_chunk = itertools.islice(pairs, start, start + chunk_size)
            cost_data = Parallel(n_jobs=parallel)(  # , backend="threading")(
                delayed(_compute_dist_cost)(pair, self.resolution)
                for pair in tqdm(
                    pairs_chunk, desc="pair", leave=False, total=chunk_size
                )
            )
            for cost in tqdm(cost_data, desc="adding edges"):
                if np.isfinite(cost[-1]):
                    G.add_edge(cost[0], cost[1], dist_cost=cost[-1])

        print(f"{len(G.edges)} edges")

    def _line_int_zero(self, state1: int, state2: int):
        return 0

    def _line_int(self, state1: int, state2: int = None, pt2: List = None) -> float:
        """Compute line integral of image likelihood costs between two coordinates

        Args:
            state1 (int): first state ID.
            loc2 (int): second state ID.

        Returns:
            [float]: sum of image likelihood costs
        """
        G = self.nxGraph
        loc1 = G.nodes[state1]["point2"]
        loc2 = G.nodes[state2]["point1"]

        return self._line_int_coord(loc1, loc2) + G.nodes[state2]["image_cost"]

    def compute_all_costs_int(self) -> None:
        """Splits up transition computation tasks then assembles them into networkx graph"""
        parallel = self.parallel
        G = self.nxGraph

        pairs = []
        for e in G.edges:
            state1_data = (e[0], G.nodes[e[0]])
            state2_data = (e[1], G.nodes[e[1]])
            pairs.append((state1_data, state2_data))

        chunk_size = 100000
        for start in tqdm(range(0, len(pairs), chunk_size), desc="pair chunks"):
            pairs_chunk = itertools.islice(pairs, start, start + chunk_size)
            cost_data = Parallel(n_jobs=parallel)(  # , backend="threading")(
                delayed(_compute_int_cost)(pair, self.tiered_path)
                for pair in tqdm(
                    pairs_chunk, desc="pair", leave=False, total=chunk_size
                )
            )
            for cost in tqdm(cost_data, desc="adding edges"):
                if np.isfinite(cost[-1]):
                    G.edges[cost[0], cost[1]]["int_cost"] = cost[-1]
                    G.edges[cost[0], cost[1]]["total_cost"] = (
                        G.edges[cost[0], cost[1]]["dist_cost"]
                        + G.edges[cost[0], cost[1]]["int_cost"]
                    )

    def shortest_path(self, coord1: List[int], coord2: List[int]) -> List[List[int]]:
        """Compute coordinate path from one coordinate to another.

        Args:
            coord1 (list): voxel coordinate of start point
            coord2 (list): voxel coordinate of end point

        Raises:
            ValueError: if state sequence contains a soma state that is not at the end

        Returns:
            list: list of voxel coordinates of path
        """
        fragments = zarr.open_array(self.fragment_path, mode="r")

        # Compute labels of coordinates
        labels = []
        for coord in [coord1, coord2]:
            local_labels, new_coord = get_valid_bbox(fragments, coord, radius=20)
            label = image_process.label_points(
                local_labels,
                [new_coord],
                res=self.resolution,
            )[1][0]
            labels.append(label)

        # find shortest path for all state combinations
        states1 = self.comp_to_states[labels[0]]
        states2 = self.comp_to_states[labels[1]]
        min_cost = -1

        for state1 in states1:
            for state2 in states2:
                try:
                    cost = nx.shortest_path_length(
                        self.nxGraph, state1, state2, weight="total_cost"
                    )
                except nx.NetworkXNoPath:
                    continue
                if cost < min_cost or min_cost == -1:
                    min_cost = cost
                    states = nx.shortest_path(
                        self.nxGraph, state1, state2, weight="total_cost"
                    )

        if min_cost == -1:
            raise nx.NetworkXNoPath(f"No path found between {coord1} and {coord2}")

        # create coordinate list
        coords = [coord1]
        if min_cost == -1:
            print("No valid path found, returning straight line")
        else:
            coords.append(list(self.nxGraph.nodes[states[0]]["point2"]))
            for i, state in enumerate(states[1:]):
                if self.nxGraph.nodes[state]["type"] == "fragment":
                    coords.append(list(self.nxGraph.nodes[state]["point1"]))
                    coords.append(list(self.nxGraph.nodes[state]["point2"]))
                elif self.nxGraph.nodes[state]["type"] == "soma":
                    coords.append(list(self.nxGraph.nodes[states[i]]["soma_pt"]))
                    if i != len(states) - 2:
                        raise ValueError("Soma state is not last state")

        coords.append(coord2)

        return coords


def explain_viterbrain(vb, c1, c2):
    # assume c1,c2 fall on a fragment
    path_coords = vb.shortest_path(c1, c2)
    comp_to_states = vb.comp_to_states
    z_frags = zarr.open_array(vb.fragment_path)

    states1 = comp_to_states[z_frags[c1[0], c1[1], c1[2]]]
    states2 = comp_to_states[z_frags[c2[0], c2[1], c2[2]]]

    min_cost = -1
    for state1 in states1:
        for state2 in states2:
            try:
                cost = nx.shortest_path_length(
                    vb.nxGraph, state1, state2, weight="total_cost"
                )
            except nx.NetworkXNoPath:
                continue
            if cost < min_cost or min_cost == -1:
                min_cost = cost
                states = nx.shortest_path(
                    vb.nxGraph, state1, state2, weight="total_cost"
                )

    print(f"{len(states)} states")
    print(f"{len(path_coords)} coordinates")

    coord_idx = 0
    for coord_idx, c in enumerate(path_coords[:-1]):
        state_idx = coord_idx // 2
        state = states[state_idx]
        if coord_idx > 0:
            prev_c = path_coords[coord_idx - 1]
            if z_frags[c[0], c[1], c[2]] != z_frags[prev_c[0], prev_c[1], prev_c[2]]:
                e = vb.nxGraph.edges[states[state_idx - 1], state]
                print(f"Transition: {states[state_idx-1]}->{state}: {e}")

        print(f"{coord_idx}: {c} f{z_frags[c[0],c[1],c[2]]} s{state}")


def get_valid_bbox(array, coord, radius):
    x1 = np.amax([coord[0] - radius, 0])
    y1 = np.amax([coord[1] - radius, 0])
    z1 = np.amax([coord[2] - radius, 0])
    x2 = np.amin([coord[0] + radius, array.shape[0]])
    y2 = np.amin([coord[1] + radius, array.shape[1]])
    z2 = np.amin([coord[2] + radius, array.shape[2]])

    subvol = np.array(np.squeeze(array[x1:x2, y1:y2, z1:z2]))
    return subvol, [coord[0] - x1, coord[1] - y1, coord[2] - z1]
