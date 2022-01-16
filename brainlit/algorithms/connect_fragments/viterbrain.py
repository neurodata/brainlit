import numpy as np
import math
import zarr
from joblib import Parallel, delayed
from tqdm import tqdm
from brainlit.viz.swc2voxel import Bresenham3D
from brainlit.preprocessing import image_process
import networkx as nx


class ViterBrain:
    def __init__(
        self,
        G,
        tiered_path,
        fragment_path,
        resolution,
        coef_curv,
        coef_dist,
        coef_int,
        parallel=1,
    ):
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

    def frag_frag_dist(self, pt1, orientation1, pt2, orientation2, verbose=False):
        """Compute cost of transition between two fragment states

        Args:
            pt1 (list of floats): first coordinate
            orientation1 (list of floats): orientation at first coordinate
            pt2 (list of floats): second coordinate
            orientation2 (list of floats): orientation at second coordinate
            verbose (bool, optional): Print transition cost information. Defaults to False.

        Raises:
            ValueError: if an orientation is not unit length
            ValueError: if distance or curvature cost is nan

        Returns:
            [float]: cost of transition
        """
        res = self.resolution

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

    def frag_soma_dist(self, point, orientation, soma_lbl, verbose=False):
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
        image_fragment = zarr.open(self.fragment_path, mode="r")

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
            cost = k_cost * self.coef_curv + self.coef_dist * (dist ** 2)

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

    def _compute_out_costs_dist(self, states, frag_frag_func, frag_soma_func):
        """Compute outgoing distance costs for specified list of states.

        Args:
            states (list of ints): list of states from which to compute transition costs.
            frag_frag_func (function): function that computes transition cost between fragments
            frag_soma_func (function): function that computes transition cost between fragments

        Raises:
            ValueError: if cannot compute transition cost between two states

        Returns:
            [list]: list of transition costs
        """
        num_states = self.num_states
        G = self.nxGraph

        results = []
        for state1 in tqdm(states, desc="computing state costs (geometry)"):
            for state2 in range(num_states):
                soma_pt = None

                if G.nodes[state1]["fragment"] == G.nodes[state2]["fragment"]:
                    continue
                elif G.nodes[state1]["type"] == "soma":
                    continue
                elif (
                    G.nodes[state1]["type"] == "fragment"
                    and G.nodes[state2]["type"] == "fragment"
                ):
                    try:
                        dist_cost = frag_frag_func(
                            G.nodes[state1]["point2"],
                            G.nodes[state1]["orientation2"],
                            G.nodes[state2]["point1"],
                            G.nodes[state2]["orientation1"],
                        )
                    except:
                        raise ValueError(
                            f"Cant compute cost between fragments: state1: {state1}, state2: {state2}, node1: {G.nodes[state1]}, node2 = {G.nodes[state2]}"
                        )
                elif (
                    G.nodes[state1]["type"] == "fragment"
                    and G.nodes[state2]["type"] == "soma"
                ):
                    dist_cost, soma_pt = frag_soma_func(
                        G.nodes[state1]["point2"],
                        G.nodes[state1]["orientation2"],
                        G.nodes[state2]["fragment"],
                    )

                if np.isfinite(dist_cost):
                    results.append((state1, state2, dist_cost, soma_pt))
        return results

    def compute_all_costs_dist(self, frag_frag_func, frag_soma_func):
        """Splits up transition computation tasks then assembles them into networkx graph

        Args:
            frag_frag_func (function): function that computes transition cost between fragments
            frag_soma_func (function): function that computes transition cost between fragments
        """
        parallel = self.parallel
        G = self.nxGraph

        state_sets = np.array_split(np.arange(self.num_states), parallel)

        results_tuple = Parallel(n_jobs=parallel)(
            delayed(self._compute_out_costs_dist)(
                states, frag_frag_func, frag_soma_func
            )
            for states in state_sets
        )

        results = [item for result in results_tuple for item in result]
        for result in results:
            state1, state2, dist_cost, soma_pt = result
            if dist_cost != np.inf:
                G.add_edge(state1, state2, dist_cost=dist_cost)
            if soma_pt is not None:
                G.nodes[state1]["soma_pt"] = soma_pt

    def _line_int(self, loc1, loc2):
        """Compute line integral of image likelihood costs between two coordinates

        Args:
            loc1 (list of ints): first coordinate
            loc2 (list of ints): second coordinate

        Returns:
            [float]: sum of image likelihood costs
        """
        image_tiered = zarr.open(self.tiered_path, mode="r")
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

    def _compute_out_int_costs(self, states):
        """Compute pairwise image likelihood costs.

        Args:
            states (list of ints): list of states

        Raises:
            ValueError: Cases did not catch the particular state type pair

        Returns:
            [list]: list of transition costs values
        """
        num_states = self.num_states
        G = self.nxGraph

        results = []
        for state1 in tqdm(states, desc="Computing state costs (intensity)"):
            for state2 in range(num_states):
                if G.nodes[state1]["fragment"] == G.nodes[state2][
                    "fragment"
                ] or not G.has_edge(state1, state2):
                    continue
                elif G.nodes[state1]["type"] == "soma":
                    continue
                elif (
                    G.nodes[state1]["type"] == "fragment"
                    and G.nodes[state2]["type"] == "fragment"
                ):
                    line_int_cost = self._line_int(
                        G.nodes[state1]["point2"], G.nodes[state2]["point1"]
                    )
                    int_cost = line_int_cost + G.nodes[state2]["image_cost"]
                    results.append((state1, state2, line_int_cost))
                elif (
                    G.nodes[state1]["type"] == "fragment"
                    and G.nodes[state2]["type"] == "soma"
                ):
                    line_int_cost = self._line_int(
                        G.nodes[state1]["point2"], G.nodes[state1]["soma_pt"]
                    )
                    results.append((state1, state2, line_int_cost))
                else:
                    raise ValueError("No cases caught int")

        return results

    def compute_all_costs_int(self):
        """Splits up transition computation tasks then assembles them into networkx graph"""
        parallel = self.parallel
        G = self.nxGraph

        state_sets = np.array_split(np.arange(self.num_states), parallel)

        results_tuple = Parallel(n_jobs=parallel)(
            delayed(self._compute_out_int_costs)(states) for states in state_sets
        )

        results = [item for result in results_tuple for item in result]
        for result in results:
            state1, state2, int_cost = result
            if int_cost != np.inf:
                G.edges[state1, state2]["int_cost"] = int_cost
                G.edges[state1, state2]["total_cost"] = (
                    G.edges[state1, state2]["int_cost"]
                    + G.edges[state1, state2]["dist_cost"]
                )

    def shortest_path(self, coord1, coord2):
        """Compute coordinate path from one coordinate to another.

        Args:
            coord1 (list): voxel coordinate of start point
            coord2 (list): voxel coordinate of end point

        Raises:
            ValueError: if state sequence contains a soma state that is not at the end

        Returns:
            list: list of voxel coordinates of path
        """
        fragments = zarr.open(self.fragment_path, mode="r")

        # compute labels of coordinates
        labels = []
        radius = 20
        for coord in [coord1, coord2]:
            local_labels = fragments[
                np.amax([coord[0] - radius, 0]) : coord[0] + radius,
                np.amax([coord[1] - radius, 0]) : coord[1] + radius,
                np.amax([coord[2] - radius, 0]) : coord[2] + radius,
            ]
            label = image_process.label_points(
                local_labels, [[radius, radius, radius]], res=self.resolution
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

        # create coordinate list
        coords = [coord1]
        coords.append(self.nxGraph.nodes[states[0]]["point2"])
        for i, state in enumerate(states[1:]):
            if self.nxGraph.nodes[state]["type"] == "fragment":
                coords.append(self.nxGraph.nodes[state]["point1"])
                coords.append(self.nxGraph.nodes[state]["point2"])
            elif self.nxGraph.nodes[state]["type"] == "soma":
                coords.append(self.nxGraph.nodes[states[i]]["soma_pt"])
                if i != len(states) - 2:
                    raise ValueError("Soma state is not last state")

        coords.append(coord2)

        return coords
