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
        coef_curv=0,
        frag_orientation_length=5,
    ):

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

        soma_locs = {}
        for soma_lbl in soma_lbls:
            labels_soma = labels == soma_lbl
            coords = np.argwhere(
                labels_soma ^ ndi.morphology.binary_erosion(labels_soma)
            )
            soma_locs[soma_lbl] = coords
        self.soma_locs = soma_locs

        # variables for emission distribution
        data_fg = self.image[self.labels > 0]
        if len(data_fg.flatten()) > 10000:
            data_sample = random.sample(list(data_fg), k=10000)
        else:
            data_sample = data_fg
        data_2d = np.expand_dims(np.sort(np.array(data_sample)), axis=1)
        kde = KernelDensity(kernel="gaussian", bandwidth=100).fit(data_2d)

        image_tiered = 0 * image
        for val in tqdm(
            np.unique(image), desc="setting up emission distribution..."
        ):
            mask = image == val
            score = kde.score_samples(np.array([[val]]))
            image_tiered[mask] = -score

        self.image_tiered = image_tiered

    def frags_to_lines(self):
        """Relies on the assumption that self.labels has values as if it came from measure.label"""
        labels = self.labels
        soma_lbls = self.soma_lbls
        comp_to_states = self.comp_to_states
        state_to_comp = self.state_to_comp
        np.amax(labels)

        int_comp_costs = {}

        for component in tqdm(
            comp_to_states.keys(), desc="Computing line representations"
        ):
            if component in soma_lbls:
                continue
            else:
                states = comp_to_states[component]

            mask = labels == component

            rmin, rmax, cmin, cmax, zmin, zmax = self.compute_bounds(mask, pad=1)
            mask = mask[rmin:rmax, cmin:cmax, zmin:zmax]

            skel = morphology.skeletonize_3d(mask)

            coords_mask = np.argwhere(mask)
            coords_skel = np.argwhere(skel)

            if len(coords_skel) < 4:
                coords = coords_mask
            else:
                coords = coords_skel

            endpoints = self.endpoints_from_coords_neighbors(coords)
            a = endpoints[0]
            try:
                b = endpoints[1]
            except:
                print(f"only 1 endpoint for component {component}")
                raise ValueError

            a = np.add(a, [rmin, cmin, zmin])
            b = np.add(b, [rmin, cmin, zmin])
            dif = b-a
            dif = dif/np.linalg.norm(dif)

            state_to_comp[states[0]][2]["coord1"] = a
            state_to_comp[states[0]][2][
                "orientation1"
            ] = dif  # orient along direction of fragment
            state_to_comp[states[0]][2]["coord2"] = b
            state_to_comp[states[0]][2]["orientation2"] = dif

            state_to_comp[states[1]][2]["coord1"] = b
            state_to_comp[states[1]][2]["orientation1"] = -dif
            state_to_comp[states[1]][2]["coord2"] = a
            state_to_comp[states[1]][2]["orientation2"] = -dif

            a = [int(x) for x in a]
            b = [int(x) for x in b]

            xlist, ylist, zlist = Bresenham3D(a[0], a[1], a[2], b[0], b[1], b[2])
            sum = np.sum(self.image_tiered[xlist, ylist, zlist])
            if sum < 0:
                warnings.warn(f"Negative int cost for comp {component}: {sum}")
            int_comp_costs[component] = sum

        self.int_comp_costs = int_comp_costs

    def endpoints_from_coords_neighbors(self, coords):
        res = self.res

        dims = np.multiply(np.amax(coords, axis=0) - np.amin(coords, axis=0), res)
        max_length = np.sqrt(np.sum([dim ** 2 for dim in dims]))

        r = 15
        if max_length < r:
            radius = max_length / 2
            close_enough = radius
        else:
            radius = r
            close_enough = 9

        A = radius_neighbors_graph(
            coords, radius=radius, metric="wminkowski", metric_params={"w": res}
        )
        degrees = np.squeeze(np.array(np.sum(A, axis=1).T, dtype=int))
        indices = np.argsort(degrees)
        sorted = [degrees[i] for i in indices]

        #point with fewest neighbors
        ends = [coords[indices[0], :]]
        #second endpoint is point with fewest neighbors that is not within "close_enough" of the first endpoint
        #close_enough gets smaller until a second point is found
        while len(ends) < 2:
            for coord_idx, degree in zip(indices, sorted):
                coord = coords[coord_idx, :]
                dists = np.array(
                    [np.linalg.norm(np.multiply(coord - end, res)) for end in ends]
                )
                if not any(dists < close_enough):
                    ends.append(coord)
                    break
            close_enough = close_enough / 2

        return ends

    def compute_bounds(self, label, pad):
        """compute coordinates of bounding box around a masked object, with given padding

        Args:
            label (np.array): mask of the object
            pad (float): padding around object in um

        Returns:
            [ints]: integer coordinates of bounding box
        """
        labels = self.labels
        res = self.res

        r = np.any(label, axis=(1, 2))
        c = np.any(label, axis=(0, 2))
        z = np.any(label, axis=(0, 1))
        rmin, rmax = np.where(r)[0][[0, -1]]
        rmin = np.amax((0, math.floor(rmin - pad / res[0])))
        rmax = np.amin((labels.shape[0], math.ceil(rmax + (pad + 1) / res[0])))
        cmin, cmax = np.where(c)[0][[0, -1]]
        cmin = np.amax((0, math.floor(cmin - (pad) / res[1])))
        cmax = np.amin((labels.shape[1], math.ceil(cmax + (pad + 1) / res[1])))
        zmin, zmax = np.where(z)[0][[0, -1]]
        zmin = np.amax((0, math.floor(zmin - (pad) / res[2])))
        zmax = np.amin((labels.shape[2], math.ceil(zmax + (pad + 1) / res[2])))
        return int(rmin), int(rmax), int(cmin), int(cmax), int(zmin), int(zmax)

    def compute_all_costs_dist(self, point_point_func, point_blob_func):
        self.soma_lbls

        cost_mat_dist = self.cost_mat_dist
        cost_mat_int = self.cost_mat_int
        state_to_comp = self.state_to_comp
        comp_to_states = self.comp_to_states
        num_states = self.num_states

        with tqdm(
            total=int(num_states ** 2), desc="Computing state costs (geometry)", smoothing=0.1
        ) as pbar:
            for state1 in range(num_states):
                state1_info = state_to_comp[state1]
                for state2 in range(num_states):
                    pbar.update(1)
                    state2_info = state_to_comp[state2]

                    if (
                        cost_mat_dist[state1, state2] != -np.inf
                    ):  # cost has already been computed
                        continue
                    elif (
                        state_to_comp[state1][1] == state_to_comp[state2][1]
                    ):  # states from same fragment
                        dist_cost = np.inf
                        #distances here are symmetric
                        cost_mat_dist[state1, state2] = dist_cost
                        cost_mat_dist[state2, state1] = dist_cost
                    elif (
                        state1_info[0] == "fragment"
                        and state2_info[0] == "fragment"
                    ):  # two fragments
                        dist_cost = point_point_func(
                            state1_info[2]["coord2"],
                            state1_info[2]["orientation2"],
                            state2_info[2]["coord1"],
                            state2_info[2]["orientation1"],
                        )

                        # distances here are symmetric, but need to find the opposite orientation
                        other_states1 = [
                            s
                            for s in comp_to_states[state_to_comp[state1][1]]
                            if s != state1
                        ]
                        other_states2 = [
                            s
                            for s in comp_to_states[state_to_comp[state2][1]]
                            if s != state2
                        ]

                        cost_mat_dist[state1, state2] = dist_cost
                        cost_mat_dist[
                            other_states2[0], other_states1[0]
                        ] = dist_cost

                    elif state1_info[0] == "soma" and state2_info[0] == "soma":
                        dist_cost == np.inf
                        #distances here are symmetric
                        cost_mat_dist[state1, state2] = dist_cost
                        cost_mat_dist[state2, state1] = dist_cost
                    elif state1_info[0] == "fragment" and state2_info[0] == "soma":
                        soma_info = state2_info
                        fragment_info = state1_info
                        fragment_state = state1
                        soma_state = state2

                        dist_cost, soma_pt = point_blob_func(
                            fragment_info[2]["coord2"],
                            fragment_info[2]["orientation2"],
                            soma_info[1],
                        )
                        self.state_to_comp[fragment_state][2][
                            "soma connection point"
                        ] = soma_pt

                        cost_mat_dist[fragment_state, soma_state] = dist_cost
                        cost_mat_dist[soma_state, fragment_state] = np.inf
                    elif state1_info[0] == "soma" and state2_info[0] == "fragment":
                        soma_info = state1_info
                        fragment_info = state2_info
                        fragment_state = state2
                        soma_state = state1

                        dist_cost, soma_pt = point_blob_func(
                            fragment_info[2]["coord2"],
                            fragment_info[2]["orientation2"],
                            soma_info[1],
                        )
                        self.state_to_comp[fragment_state][2][
                            "soma connection point"
                        ] = soma_pt
                        
                        cost_mat_dist[fragment_state, soma_state] = dist_cost
                        cost_mat_dist[soma_state, fragment_state] = np.inf
                    else:
                        raise ValueError("No cases caught dist")

        # normalize dist mat
        for state1 in tqdm(range(num_states), desc="Normalizing"):
            state1_info = state_to_comp[state1]
            if state1_info[0] == "soma":
                denom = 0
            else:
                denom = logsumexp(-1 * cost_mat_dist[state1, :])
            cost_mat_dist[state1, :] = cost_mat_dist[state1, :] + denom

    def point_point_dist(self, pt1, orientation1, pt2, orientation2, verbose=False):
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
        if 1-k1_sq < -0.87 or 1-k2_sq < -0.87:
            return np.inf

        cost = k_cost * self.coef_curv + self.coef_dist * (dist ** 2)
        if verbose:
            print(
                f"Distance: {dist}, Curv penalty: {k_cost} (dots {1-k1_sq}, {1-k2_sq}, from dif-{dif}), Total cost: {cost}"
            )

        return cost

    def point_blob_dist(self, point, orientation, blob_lbl, verbose=False):
        """Assumes that the first label has endpoints (line mode)"""
        labels = self.labels
        soma_locs = self.soma_locs

        coords = soma_locs[blob_lbl]
        difs = np.multiply(np.subtract(coords, point), self.res)
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
            labels[
                nonline_point[0],
                nonline_point[1],
                nonline_point[2],
            ]
            != blob_lbl
        ):
            raise ValueError("Error in setting connection_mat")
        if verbose:
            print(
                f"Distance: {dist}, Curv penalty: {k_cost}, Total cost: {cost}, connection point: {nonline_point}"
            )

        return cost, nonline_point

    def compute_all_costs_int(self):
        #should be run after compute all dist costs
        cost_mat_dist = self.cost_mat_dist
        cost_mat_int = self.cost_mat_int
        state_to_comp = self.state_to_comp
        comp_to_states = self.comp_to_states
        num_states = self.num_states

        with tqdm(
            total=int(num_states ** 2), desc="Computing state costs (intensity)", smoothing=0.1
        ) as pbar:
            for state1 in range(num_states):
                state1_info = state_to_comp[state1]
                for state2 in range(num_states):
                    pbar.update(1)

                    state2_info = state_to_comp[state2]
                    if (
                        cost_mat_int[state1, state2] != -np.inf
                    ):  # cost has already been computed or distance cost is infinite
                        continue
                    elif (
                        state_to_comp[state1][1] == state_to_comp[state2][1]
                        or cost_mat_dist[state1, state2] == np.inf
                    ):  # states from same fragment or distance is infinite
                        int_cost = np.inf
                        # costs are not necessarily symmetric here (cost mat dist case)
                        cost_mat_int[state1, state2] != int_cost
                    elif (
                        state1_info[0] == "fragment"
                        and state2_info[0] == "fragment"
                    ):  # two fragments
                        line_int_cost = self.line_int(
                            state1_info[2]["coord2"], state2_info[2]["coord1"]
                        )
                        
                        # distances here are symmetric, but need to find the opposite orientation
                        other_states1 = [
                            s
                            for s in comp_to_states[state_to_comp[state1][1]]
                            if s != state1
                        ]
                        other_states2 = [
                            s
                            for s in comp_to_states[state_to_comp[state2][1]]
                            if s != state2
                        ]

                        cost_mat_int[other_states2[0], other_states1[0]] = self.int_comp_costs[state1_info[1]] + line_int_cost
                        cost_mat_int[state1, state2] = self.int_comp_costs[state2_info[1]] + line_int_cost
                    elif state1_info[0] == "fragment" and state2_info[0] == "soma":
                        soma_info = state2_info
                        fragment_info = state1_info
                        fragment_state = state1
                        soma_state = state2

                        try:
                            int_cost = self.line_int(
                                    fragment_info[2]["coord2"],
                                    state_to_comp[fragment_state][2][
                                        "soma connection point"
                                    ],
                                )
                        except:
                            c1 = fragment_info[2]["coord2"]
                            c2 = state_to_comp[fragment_state][2]["soma connection point"]
                            raise ValueError(f"state {state1} to {state2}, coord {c1} to {c2}")
                        cost_mat_int[fragment_state, soma_state] = int_cost   
                        cost_mat_int[soma_state, fragment_state] = np.inf  
                    elif state1_info[0] == "soma" and state2_info[0] == "fragment":
                        soma_info = state1_info
                        fragment_info = state2_info
                        fragment_state = state2
                        soma_state = state1

                        int_cost = self.line_int(
                                fragment_info[2]["coord2"],
                                state_to_comp[fragment_state][2][
                                    "soma connection point"
                                ],
                            )
                        cost_mat_int[fragment_state, soma_state] = int_cost 
                        cost_mat_int[soma_state, fragment_state] = np.inf 
                    else:
                        raise ValueError("No cases caught int")

    def line_int(self, loc1, loc2):
        """Compute an observable cost based on line between two points

        Args:
            loc1 (np.array): voxel coordinates of one point
            loc2 (np.array): voxel coordinates of another point

        Returns:
            float: cost of intensity between two states
        """
        image_tiered = self.image_tiered

        loc1 = [int(x) for x in loc1]
        loc2 = [int(x) for x in loc2]

        xlist, ylist, zlist = Bresenham3D(
            loc1[0], loc1[1], loc1[2], loc2[0], loc2[1], loc2[2]
        )
        # exclude first and last points because they are included in the component intensity sum
        xlist = xlist[1:-1]
        ylist = ylist[1:-1]
        zlist = zlist[1:-1]

        sum = np.sum(image_tiered[xlist, ylist, zlist])

        return sum

    def reset_dists(self, type="all"):
        num_states = self.num_states
        if type == "dist" or type == "all":
            self.cost_mat_dist = np.ones((num_states, num_states)) * -np.inf
        if type == "int" or type == "all":
            self.cost_mat_int = np.ones((num_states, num_states)) * -np.inf

    def create_nx_graph(self):
        nxGraph = nx.DiGraph()
        state_to_comp = self.state_to_comp
        for state in tqdm(state_to_comp.keys(), desc="Adding nodes to nx graph"):
            attr_dict = {}
            attr_dict["comp"] = state_to_comp[state][1]
            attr_dict["type"] = state_to_comp[state][0]
            if state_to_comp[state][0] == "fragment":
                keys = [
                    "coord1",
                    "orientation1",
                    "coord2",
                    "orientation2",
                    "soma connection point",
                ]
                for key in keys:
                    attr_dict[key] = state_to_comp[state][2][key]
            nxGraph.add_node(state, attr_dict=attr_dict)

        for row_num, row in enumerate(
            tqdm(self.cost_mat_dist, desc="Adding edges to nx graph")
        ):
            for col_num, col in enumerate(row):
                w = col + self.cost_mat_int[row_num, col_num]
                if np.isfinite(w):
                    nxGraph.add_edge(row_num, col_num, weight=w)
        self.nxGraph = nxGraph

