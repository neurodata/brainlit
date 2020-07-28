#%%
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize
from sklearn.feature_extraction.image import grid_to_graph
import scipy.ndimage as ndi
from scipy import optimize
from scipy.interpolate import splrep, splprep, interp1d
from ...viz import swc2voxel
import cv2
import math
import warnings
import networkx as nx
from joblib import Parallel, delayed
import warnings
from itertools import product

# import quadprog
import pickle
import warnings


class Connections:
    def __init__(self, image_seg, image_int, voxel_dims=[1, 1, 1], verbose=False):
        self.image = image_seg
        self.image_int = image_int
        self.image_labeled, self.num_comp = label(self.image, return_num=True)
        self.image_skel = skeletonize(self.image, method="lee")
        self.voxel_dims = np.array(voxel_dims)
        self.coms = None
        self.locs = None
        self.lines = None
        self.skels = None
        self.graphs = None
        self.verbose = verbose
        # coordinates at location i,j will be coordinate in component i,
        # closest to j
        self.closest_idxs = -1 * np.ones((self.num_comp, self.num_comp, 3))

    def compute_locs(self):
        if self.locs is None:
            self.locs = []
            for lab in range(1, self.num_comp + 1):
                loc = np.where(self.image_labeled == lab)
                loc = np.stack(loc, axis=1)
                self.locs.append(loc)

        return self.locs

    def compute_coms(self):
        if self.coms is None:
            self.coms = ndi.measurements.center_of_mass(
                self.image, self.image_labeled, [i for i in range(1, self.num_comp + 1)]
            )

        return self.coms

    def compute_lines(self):
        if self.lines is None:
            if self.locs is None:
                self.compute_locs()

            self.lines = []
            for loc in self.locs:
                line = cv2.fitLine(
                    loc, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01
                )
                vec = np.squeeze(np.array(line[0:3]))
                self.lines.append(vec)

        return self.lines

    def compute_skels(self):
        """
        Important notes about the functions used here:

        skeletonize - diagonals count as neighbors
        e.g.

        a = np.array([[0., 0., 0., 1., 1.],
           [0., 0., 1., 1., 1.],
           [0., 1., 1., 1., 0.],
           [1., 1., 1., 0., 0.],
           [1., 1., 0., 0., 0.]])

           will give a skeleton going up the diagonal
        """
        if self.skels is None:
            self.skels = []
            decayed = 0 * self.image_labeled
            decayed_num = 0
            for lab in range(1, self.num_comp + 1):
                im_1comp = self.image_labeled == lab
                skel = (self.image_skel > 0) & im_1comp
                loc = np.where(skel)
                loc = np.stack(loc, axis=1)
                if loc.shape[0] == 0:
                    decayed = decayed + im_1comp
                    decayed_num = decayed_num + 1
                self.skels.append(loc)
            self.im_decayed = decayed
            if decayed_num > 0:
                msg = f"{decayed_num} out of {self.num_comp} components disappeared during skeletonization."
                warnings.warn(msg, RuntimeWarning)

        return self.skels

    def compute_graphs(self, location_attr="loc", connectivity=26):
        if self.graphs is None:
            skels = self.compute_skels()
            self.graphs = []
            self.graphs = Parallel(n_jobs=4, verbose=1, backend="threading")(
                map(delayed(self.compute_graph), skels)
            )
            # for s,skel in enumerate(skels):
            #    print(f'{s} out of {len(skels)}')
            #    print(f'{skel.shape[0]} points')
            #    g = self.compute_graph(skel, location_attr=location_attr, connectivity=connectivity)
            #    self.graphs.append(g)

        return self.graphs

    def compute_graph(self, skel, location_attr="loc", connectivity=26):
        sz = self.image.shape

        mask = self.image_labeled * 0
        if skel.shape[0] == 0:
            return None

        mask[skel[:, 0], skel[:, 1], skel[:, 2]] = 1
        locs_flattened = skel[:, 0] * sz[0] * sz[1] + skel[:, 1] * sz[0] + skel[:, 2]
        sorted_idxs = np.argsort(locs_flattened)
        coords_sorted = skel[sorted_idxs, :]
        graph = grid_to_graph(
            self.image.shape[0],
            self.image.shape[1],
            self.image.shape[2],
            mask=mask,
            connectivity=connectivity,
        )
        g = GeometricGraph()
        nodes = [
            (
                node_id,
                {
                    location_attr: coord,
                    "intensity": self.image_int[coord[0], coord[1], coord[2]],
                },
            )
            for node_id, coord in enumerate(coords_sorted)
        ]

        g.add_nodes_from(nodes)

        edges = [(a, b) for a, b in zip(graph.row, graph.col) if a > b]
        g.add_edges_from(edges)

        if not nx.is_tree(g):
            g.remove_cycles()
            print("*Converted to Tree?**")
            print(nx.is_tree(g))
            print(len(g.nodes))

        return g

    def l2(self, pt1, pt2):
        return np.sqrt(np.sum(np.square(self.voxel_dims * np.absolute(pt1 - pt2))))

    def dist_l2(self, idx1, idx2):
        if self.coms is None:
            self.compute_coms()

        d = np.inf

        com1 = np.array(self.coms[idx1])
        com2 = np.array(self.coms[idx2])

        d = self.l2(com1, com2)

        return d

    def dist_l2_closest(self, idx1, idx2):
        self.compute_skels()
        if self.verbose:
            print(f"Connecting #{idx1} and #{idx2}")
            print(f"{len(self.skels[idx1])} and {len(self.skels[idx2])}")

        d = np.inf

        for loc1 in self.skels[idx1]:
            for loc2 in self.skels[idx2]:
                if self.l2(loc1, loc2) < d:
                    d = self.l2(loc1, loc2)
                    self.closest_idxs[idx1, idx2, :] = loc1
                    self.closest_idxs[idx2, idx1, :] = loc2

        return d

    def dist_angle(self, idx1, idx2):
        self.compute_lines()

        d = np.inf
        dotmag = np.absolute(np.dot(self.lines[idx1], self.lines[idx2]))
        d = math.acos(dotmag)
        return d

    def dist_l2_closest_constrained(self, idx1, idx2):
        if self.dist_angle(idx1, idx2) > 0.8:
            return np.inf
        else:
            return self.dist_l2_closest(idx1, idx2)

    def dist_graphs(self, idx1, idx2):
        """Custom multi-component dissimilarity between the graph forms
        of two connected components. Finds the minimum dissimilarity between
        all pairs of nodes in the two graphs.

        Arguments:
            idx1 {int} -- index of first component
            idx2 {int} -- index of second component

        Returns:
            float -- dissimilarity (>= 0)
        """
        print(f"Computing distance between {idx1} and {idx2}")
        graphs = self.compute_graphs()

        d = np.inf

        if graphs[idx1] is None or graphs[idx2] is None:
            return d

        d, n1, n2 = dist_graphs_general(graphs[idx1], graphs[idx2])
        self.closest_idxs[idx1, idx2, :] = graphs[idx1].nodes[n1]["loc"]
        self.closest_idxs[idx2, idx1, :] = graphs[idx2].nodes[n2]["loc"]

        return d

    def compute_distmat(self):
        self.W = np.zeros((self.num_comp, self.num_comp))
        for i in range(self.num_comp):
            for j in range(self.num_comp):
                self.W[i, j] = self.dist_l2(i, j)

        return self.W

    def Primms(self, dist):
        N = self.num_comp
        edges = [(i, j) for i in range(N) for j in range(N) if i < j]

        # cost of minimum connection (starts as constant, computed later)
        Cv = np.ones(N) * 1e10

        # index of minimum cost edge (-1 is special flag value meaning it has not yet been computed)
        Ev = [-1 for i in range(N)]
        # empty forest
        Fv = []
        Fe = []
        # vertices not yet in forest
        Q = np.arange(N)

        # repeat while Q is not empty
        while np.any(Cv < np.inf):
            # find minimum cost
            ind = np.argmin(Cv)

            # add vertex to forest
            Fv.append(Q[ind])

            # is Ev the flag value? (have its connection weights been computed)
            if Ev[ind] != -1:
                # if not, it is the cheapest addition - add it
                Fe.append(Ev[ind])

            # now loop over all fragments connected to this one
            # if the fragment has not been added yet, AND has a smaller weight than Cv
            # update Cv, and update Ev
            for e in edges:
                # keep looping if it's not this fragment
                if not (e[0] == ind or e[1] == ind):
                    continue

                # make sure e0 points to newly added vertex
                # e1 points to the other vertex whose cost I adjust
                if e[1] == ind:
                    e = e[::-1]

                # keep looping if this vertex has already been added
                if Cv[e[1]] == np.inf:
                    continue

                # otherwise we need to update Cv if it has a lower cost
                this_frag = e[0]
                if this_frag == ind:
                    this_frag = e[1]
                Cv_ = dist(ind, this_frag)
                if Cv_ < Cv[this_frag]:
                    Cv[this_frag] = Cv_
                    Ev[this_frag] = e

            # make sure I never pick this vertex again
            Cv[ind] = np.inf
        return Fe

    def draw_connections(self, alg, dist, connection_pts="com", pctile_tresh=90):
        edges = alg(dist)

        edge_weights = np.zeros((len(edges)))
        for idx, e in enumerate(edges):
            edge_weights[idx] = dist(e[0], e[1])
        thresh = np.percentile(edge_weights, pctile_tresh)

        # Can be deleted when you stop using the pickled connection object
        self.image_skel = skeletonize(self.image, method="lee")

        self.connected_image = self.image_skel.copy()
        maxval = np.amax(self.connected_image)

        if connection_pts == "com":
            self.compute_coms()

        for e in edges:
            edge_weight = dist(e[0], e[1])
            if edge_weight < thresh:
                if connection_pts == "com":
                    p0 = np.round(np.array(self.coms[e[0]]))
                    p1 = np.round(np.array(self.coms[e[1]]))
                elif connection_pts == "closest":
                    p0 = self.closest_idxs[e[0], e[1], :]
                    p1 = self.closest_idxs[e[1], e[0], :]

                xs, ys, zs = swc2voxel.Bresenham3D(
                    p0[0], p0[1], p0[2], p1[0], p1[1], p1[2]
                )
                xs = np.array(xs, dtype=int)
                ys = np.array(ys, dtype=int)
                zs = np.array(zs, dtype=int)
                self.connected_image[xs, ys, zs] = maxval

        return self.connected_image

    def draw_downsampled_pts(self, period):
        self.compute_graphs()

        self.downsampled_img = 0 * self.image_labeled

        for g in self.graphs:
            if g is None:
                continue
            elif nx.is_tree(g):
                g_downsampled = g.downsample(period=period)

                for n in g_downsampled.nodes:
                    loc = g_downsampled.nodes[n]["loc"]
                    self.downsampled_img[loc[0], loc[1], loc[2]] = 1

        return self.downsampled_img

    def remove_blobs(self):
        locs = self.compute_locs()
        im_blobs_removed = (0 * self.image).astype(bool)
        for loc in locs:
            fill = cube_fill(loc)
            if fill < 0.1:
                im_blobs_removed[np.split(loc.T, loc.T.shape[0], axis=0)] = True
        return im_blobs_removed


"""
Function definitions
"""


def cube_fill(loc_list):
    mins = np.amin(loc_list, axis=0)
    lengths = np.amax(loc_list, axis=0) - mins + 1
    vol = np.amax(lengths) ** 3
    fill = loc_list.shape[0]

    if fill / vol > 1:
        raise ValueError

    return fill / vol


def dist_graphs_general(g1, g2, locattr="loc", intattr="intensity"):
    """Compute 3 component dissimilarity between graph objects.
        Nodes in the graphs must have the attributes locattr and intattr

    Arguments:
        g1 {GeometricGraph} -- graph #1
        g2 {GeometricGraph} -- graph #2
        locattr {string} -- name of location attribute in the graph nodes
        intattr {string} -- name of intensity attribute in the graph nodes

    Returns:
        tuple -- distance between the graphs, and the "closest" pair of nodes
    """
    d = np.inf
    for n1 in g1.nodes:
        for n2 in g2.nodes:
            node1 = (g1, n1)
            node2 = (g2, n2)
            d_temp = dist_vertices_general(node1, node2, locattr, intattr)

            if d_temp < d:
                d = d_temp
                best_n1 = n1
                best_n2 = n2

    return (d, best_n1, best_n2)


def dist_vertices_general(
    node1,
    node2,
    locattr="loc",
    intattr="intensity",
    weights=[1, 1, 1],
    normalization=[12 * (10 ** 6), 30000 ** 2, 8],
):
    """Compute 3 component dissimilarity between two nodes of different GeometricGraph objects

    Arguments:
        node1 {tuple} -- graph1 and the node in question
        node2 {tuple} -- graph2 and the node in question

    Keyword Arguments:
        locattr {str} -- name of the node attribute that contains the spatial location (default: {"loc"})
        intattr {str} -- name of the node attribute that contains the image intensity (default: {"intensity"})
        weights {list} -- weights on each component of the dissimilarity (default: {[1, 1, 1]})
        normalization {list} -- normalization of each component of the dissimilarity (default: {[12 * (10 ** 6), 30000 ** 2, 8]})

    Raises:
        ValueError: [description]
        ValueError: [description]

    Returns:
        float -- dissimilarity between the two nodes
    """
    g1 = node1[0]
    g2 = node2[0]

    # Component 1) euclidean distance in space
    loc_1 = g1.nodes[node1[1]][locattr]
    loc_2 = g2.nodes[node2[1]][locattr]
    d_space = np.linalg.norm(loc_1 - loc_2)

    # Component 2) squared intensity difference in image
    b_1 = g1.nodes[node1[1]][intattr]
    b_2 = g2.nodes[node2[1]][intattr]
    d_brightness = (b_1 - b_2) ** 2

    # Component 3) curvatures from connections
    # get list of all possible directions after connection
    segments_1 = nx.dfs_tree(g1, source=node1[1], depth_limit=3)
    # enumerate the possible paths after connection
    paths1 = get_paths_to_leaves(segments_1, node1[1])
    # convert paths into arrays of locations
    locmats1 = convert_paths_to_locmats(g1, paths1)
    # repeat on other fragment
    segments_2 = nx.dfs_tree(g2, source=node2[1], depth_limit=3)
    paths2 = get_paths_to_leaves(segments_2, node2[1])
    locmats2 = convert_paths_to_locmats(g2, paths2)

    d_curv = np.inf

    # for each possible path of connection
    for locmat1, locmat2 in product(locmats1, locmats2):
        half1 = np.flip(locmat1, axis=0)
        connection_point = half1.shape[0]
        half2 = locmat2
        # list of path locations
        locmat = np.concatenate((half1, half2), axis=0)
        # compute curvature
        curv, tor = GeometricGraph.compute_curv_tor(locmat)
        # ssquared sum of curvatures at both connection points
        d_curv_temp = curvature_cost(
            curv[connection_point - 1], curv[connection_point], normalization[2]
        )
        if d_curv_temp < d_curv:
            d_curv = d_curv_temp

    d = [
        d_space / normalization[0],
        d_brightness / normalization[1],
        d_curv,
    ]

    return np.dot(d, weights)


# with open("/cis/home/tathey/projects/mouselight/large_files/curvatures.curvs", 'rb') as curv_file:
#    curvs = pickle.load(curv_file)


def curvature_cost(curv1, curv2, normalization, mode="sq"):
    if mode == "sq":
        return (curv1 ** 2 + curv2 ** 2) / normalization
    elif mode == "cdf":
        prob = (np.sum(curvs < curv1) + np.sum(curvs < curv2)) / (2 * len(curvs))
        return prob


def convert_paths_to_locmats(graph, paths):
    """Convert paths to array of locations

    Arguments:
        graph {nx.Graph} -- graph with location info of paths (e.g. connected component)
        paths {list of lists} -- list of paths

    Returns:
        list of nd.arrays -- list of arrays that hold the spatial locations of the paths
    """
    locmats = []
    for path in paths:
        path_length = len(path)
        locmat = np.zeros([path_length, 3])
        for i, node in enumerate(path):
            locmat[i, :] = graph.nodes[node]["loc"]

        locmats.append(locmat)
    return locmats


def get_paths_to_leaves(digraph, root):
    """Make list of paths from root to leaves in a directed graph

    Arguments:
        digraph {nx.Digraph} -- directed graph (e.g. depth first traversal
                                from bridge node candidate)
        root {nx.Node} -- root node (e.g. bridge node candidate)

    Returns:
        list of lists -- paths from root to all leavesd
    """
    paths = []
    for node in digraph:
        if digraph.out_degree(node) == 0:
            paths.append(nx.shortest_path(digraph, root, node))

    return paths


def checkIfDuplicates_2(listOfElems):
    """ Check if given list contains any duplicates """
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False


"""
Geometric Graph class
"""


class GeometricGraph(nx.Graph):
    def __init__(self):
        super(GeometricGraph, self).__init__()
        self.segments = None
        self.cycle = None
        self.root = 1

    def fit_dataframe(self, df):
        """
        Fits a GeometricGraph object to a dataframe by adding each node row by
        row, and adding edges between nodes and their parents.

        Parameters
        ----------
        df : dataframe
            the axon object
        """
        num_pts = df.shape[0]
        first = True
        for row in df.itertuples(index=False):
            samp = row.sample
            loc = np.array([row.x, row.y, row.z])
            if first:
                soma = loc
                first = False
            self.add_node(samp, loc=loc)
            par = row.parent
            if par != -1:
                if par > samp:
                    raise ValueError("Parent has not been added yet")
                self.add_edge(samp, par)

    def set_root(self, root):
        self.root_node = root

    def is_cyclic(self):
        try:
            self.cycle = nx.find_cycle(self)
            return True
        except nx.exception.NetworkXNoCycle:
            return False

    def is_list(self):
        degs = list(self.degree(self.nodes))
        for deg in degs:
            if deg[1] > 2:
                return False

        return True

    def remove_cycles(self):
        """
        Note: this may turn a 2d circle into a tree
        e.g.
        0 1 1 0
        1 0 0 1
        0 1 1 0

        will be trimmed at a corner.

        However, I do not think that trimming here will lose connectedness.
        """
        try:
            while True:
                cycle = nx.find_cycle(self)
                trimmed = self.trim_cycle_18(cycle)
                if not trimmed:
                    trimmed = self.trim_cycle_6(cycle)
                    if not trimmed:
                        return False
        except nx.exception.NetworkXNoCycle:
            return True

    def trim_cycle_6(self, cycle):
        print("***Trimming**")
        for e in cycle:
            loc1 = self.nodes[e[0]]["loc"]
            loc2 = self.nodes[e[1]]["loc"]
            if sum([i == j for i, j in zip(loc1, loc2)]) <= 1:
                self.remove_edge(e[0], e[1])
                return True
        return False

    def trim_cycle_18(self, cycle):
        for e in cycle:
            loc1 = self.nodes[e[0]]["loc"]
            loc2 = self.nodes[e[1]]["loc"]
            if sum([i == j for i, j in zip(loc1, loc2)]) == 0:
                self.remove_edge(e[0], e[1])
                return True
        return False

    def downsample(self, period, voxel_size=[1, 1, 1], location_attr="loc"):
        if not nx.is_tree(self):
            raise ValueError
        if len(self.nodes) == 1:
            return self.copy()

        downsampled_graph = GeometricGraph()

        segments = self.split_segments()

        for segment in segments:
            downsampled_segment = self.downsample_segment(
                segment, period=period, voxel_size=voxel_size
            )

            for i, node in enumerate(downsampled_segment):
                downsampled_graph.add_node(node, loc=self.nodes[node]["loc"])
                if i != 0:
                    downsampled_graph.add_edge(
                        downsampled_segment[i], downsampled_segment[i - 1]
                    )
        return downsampled_graph

    def split_segments(self):
        if len(self.nodes()) <= 1:
            raise ValueError("Cannot split graph with <= 1 nodes")

        if self.segments is None:
            for n in self.nodes():
                if self.degree(n) == 1:
                    leaf = n
                    break

            traversal = nx.algorithms.traversal.depth_first_search.dfs_preorder_nodes(
                self, source=leaf
            )

            start_stack = []
            segments = []
            segment = []

            for i, n in enumerate(traversal):
                if i == 0:
                    start_stack.append(n)
                    continue

                deg = self.degree(n)

                if deg == 2:
                    segment.append(n)
                else:
                    segment.append(n)
                    start = start_stack.pop()
                    segment.insert(0, start)
                    segments.append(segment)
                    segment = []

                    if deg > 2:
                        for i in range(deg - 1):
                            start_stack.append(n)
                self.segments = segments

        return self.segments

    def downsample_segment(self, segment, period, voxel_size=[1, 1, 1]):
        downsampled = []
        # compute complete length
        length = 0
        for i in range(len(segment) - 1):
            n1 = segment[i]
            pt1 = self.nodes[n1]["loc"]
            n2 = segment[i + 1]
            pt2 = self.nodes[n2]["loc"]
            length = length + np.linalg.norm(np.multiply(pt1 - pt2, voxel_size))

        num_segs = np.ceil(length / period)
        seg_len = length / num_segs

        downsampled.append(segment[0])

        length = 0
        seg_num = 1
        for i in range(1, len(segment) - 1):
            n1 = segment[i - 1]
            pt1 = self.nodes[n1]["loc"]
            n2 = segment[i]
            pt2 = self.nodes[n2]["loc"]
            length = length + np.linalg.norm(np.multiply(pt1 - pt2, voxel_size))
            if length >= seg_num * seg_len:
                downsampled.append(segment[i])
                seg_num = seg_num + 1

        downsampled.append(segment[-1])
        return downsampled

    def param_segs(self, smooth=False):
        loc_mats = self.compute_locmats()
        self.curvatures = []
        self.torsions = []
        for x in loc_mats:
            curv, tor = self.compute_curv_tor(x, smooth=smooth)
            self.curvatures.append(curv)
            self.torsions.append(tor)

        return self.curvatures, self.torsions

    @staticmethod
    def compute_curv_tor(x, spacing=1, init_conds=False, smooth=False):
        if smooth:
            x0 = ndi.gaussian_filter(x[:, 0], 2)
            x1 = ndi.gaussian_filter(x[:, 1], 2)
            x2 = ndi.gaussian_filter(x[:, 2], 2)
            x = np.stack((x0, x1, x2), axis=1)
        # Definition 7.37 in Grenander, Miller Pattern Theory
        dx_dt = np.gradient(x, spacing, axis=0)
        a = np.linalg.norm(dx_dt, axis=1)
        dx_dt2 = np.gradient(dx_dt, spacing, axis=0)
        cross = np.cross(dx_dt, dx_dt2, axisa=1, axisb=1)
        k = np.divide(np.linalg.norm(cross, axis=1), a ** 3)
        dx_dt3 = np.gradient(dx_dt2, spacing, axis=0)

        t = np.divide(
            np.sum(np.multiply(cross, dx_dt3), axis=1),
            np.linalg.norm(cross, axis=1) ** 2,
        )
        # t[np.linalg.norm(cross, axis=1) == 0] = 0

        if init_conds:
            T0 = dx_dt[0, :] / np.linalg.norm(dx_dt[0, :])
            N0 = np.array([0, 1, 0])
            for i in np.arange(dx_dt2.shape[0]):
                if np.linalg.norm(dx_dt2[i, :]) != 0:
                    N0 = dx_dt2[i, :] / np.linalg.norm(dx_dt2[i, :])
                    break
            B0 = np.cross(T0, N0)
            return k, t, (T0, N0, B0)
        else:
            return k, t

    def Frenet_segs(self):
        loc_mats = self.compute_locmats()
        for loc_mat in loc_mats:
            T, N, B = self.Frenet(points=loc_mat)

    def Frenet(self, points):
        # https://janakiev.com/blog/framing-parametric-curves/
        # Number of points
        n = len(points)

        # Calculate the first and second derivative of the points
        dX = np.apply_along_axis(np.gradient, axis=0, arr=points)
        ddX = np.apply_along_axis(np.gradient, axis=0, arr=dX)

        # Normalize all tangents
        f = lambda m: m / np.linalg.norm(m)
        T = np.apply_along_axis(f, axis=1, arr=dX)

        # Calculate and normalize all binormals
        B = np.cross(dX, ddX)
        B = np.apply_along_axis(f, axis=1, arr=B)

        # Calculate all normals
        N = np.cross(B, T)

        return T, N, B

    def compute_locmats(self):
        if self.segments is None:
            self.segments = self.split_segments()

        loc_mats = []

        for segment in self.segments:
            loc_mat = np.zeros([len(segment), 3])  # , dtype=int)
            for i, n in enumerate(segment):
                loc_mat[i, :] = self.nodes[n]["loc"]
            loc_mats.append(loc_mat)

        return loc_mats

    def fit_curves(self, curve_type="poly", *args):
        loc_mats = self.compute_locmats()

        self.curves = []
        for y in loc_mats:
            x = np.arange(y.shape[0])

            if curve_type == "poly":
                curve = np.polyfit(x, y, deg=args[0])
                self.curves.append((y.shape[0], curve))
            elif curve_type == "predef":
                popt_x, _ = optimize.curve_fit(args[0], x, y[:, 0])
                popt_y, _ = optimize.curve_fit(args[0], x, y[:, 1])
                popt_z, _ = optimize.curve_fit(args[0], x, y[:, 2])
                self.curves.append((y.shape[0], popt_x, popt_y, popt_z))
            elif curve_type == "b":
                try:
                    k = np.amin([len(x) - 1, 3])
                    tckx = splrep(x, y[:, 0], k=k, s=args[0])
                    tcky = splrep(x, y[:, 1], k=k, s=args[0])
                    tckz = splrep(x, y[:, 2], k=k, s=args[0])
                    self.curves.append((y.shape[0], tckx, tcky, tckz))
                except TypeError:
                    print(f"Type Error for sequence of length: {len(x)}")
                    self.curves.append(None)
            elif curve_type == "interp":
                interpx = interp1d(x, y[:, 0], kind=args[0])
                interpy = interp1d(x, y[:, 1], kind=args[0])
                interpz = interp1d(x, y[:, 2], kind=args[0])
                self.curves.append((y.shape[0], interpx, interpy, interpz))

        return self.curves

    def get_ends(self):
        ends = []
        for n in self.nodes:
            d = self.degree(n)
            if d == 1:
                ends.append(n)

        return ends

    def get_end_coords(self):
        ends = self.get_ends()
        locmat = np.zeros((len(ends), 3))

        for i, end in enumerate(ends):
            locmat[i, :] = self.nodes[end]["loc"]

        return locmat

    def fit_spline_tree_invariant(self):
        spline_tree = nx.DiGraph()
        curr_spline_num = 0
        stack = []
        root = self.root
        tree = nx.algorithms.traversal.depth_first_search.dfs_tree(self, source=root)

        path, other_trees = self.find_longest_path(tree)
        spline_tree.add_node(curr_spline_num, path=path, starting_length=0)

        for tree in other_trees:
            stack.append((tree, curr_spline_num))

        while len(stack) > 0:
            curr_spline_num = curr_spline_num + 1
            treenum = stack.pop()
            tree = treenum[0]
            parent_num = treenum[1]

            path, other_trees = self.find_longest_path(tree[0], starting_length=tree[2])
            path.insert(0, tree[1])

            spline_tree.add_node(curr_spline_num, path=path, starting_length=tree[2])
            spline_tree.add_edge(parent_num, curr_spline_num)

            for tree in other_trees:
                stack.append((tree, curr_spline_num))

        for node in spline_tree.nodes:
            path = spline_tree.nodes[node]["path"]

            spline_tree.nodes[node]["spline"] = self.fit_spline_path(path)

        return spline_tree

    def fit_spline_path(self, path):
        x = np.zeros((len(path), 3))

        for row, node in enumerate(path):
            x[row, :] = self.nodes[node]["loc"]
        orig = x.shape[0]
        x = [xi for i, xi in enumerate(x) if i == 0 or (xi != x[i - 1, :]).any()]
        x = np.stack(x, axis=0)
        new = x.shape[0]
        if orig != new:
            warnings.warn(
                f"{orig-new} duplicate points removed in the trace segment",
                category=UserWarning,
            )
        m = x.shape[0]
        diffs = np.diff(x, axis=0)
        diffs = np.linalg.norm(diffs, axis=1)
        diffs = np.cumsum(diffs)
        diffs = np.concatenate(([0], diffs))
        k = np.amin([m - 1, 5])
        tck, u = splprep([x[:, 0], x[:, 1], x[:, 2]], u=diffs, k=k)

        self.check_multiplicity(tck[0])

        return tck, u

    def check_multiplicity(self, t):
        knots = list(t.copy())
        first = knots[0]
        last = knots[-1]
        indices_keep = (knots != first) & (knots != last)
        knots = [i for (i, v) in zip(knots, indices_keep) if v]
        dup = checkIfDuplicates_2(knots)
        if dup:
            print(t)
            raise RuntimeError("Duplicates found in the above knot list")

    def find_longest_path(self, tree, starting_length=0):
        other_trees = []
        if len(tree.nodes) == 1:
            path = tree.nodes
        else:
            roots = [n for n, d in tree.in_degree() if d == 0]

            if len(roots) > 1:
                raise ValueError("More than one node with in degree 0")
            else:
                root = roots[0]

            leaves = [
                n
                for n in tree.nodes()
                if tree.out_degree(n) == 0 and tree.in_degree(n) == 1
            ]

            shortest_paths = [
                nx.algorithms.shortest_paths.generic.shortest_path(
                    tree, source=root, target=l
                )
                for l in leaves
            ]

            distances = [self.path_length(path) for path in shortest_paths]
            shortest_idx = np.argmax(distances)

            leaf = leaves[shortest_idx]

            path = nx.algorithms.shortest_paths.generic.shortest_path(
                tree, source=root, target=leaf
            )

            length = starting_length
            for i, node in enumerate(path):
                if i > 0:
                    loc1 = self.nodes[node]["loc"]
                    loc2 = self.nodes[path[i - 1]]["loc"]
                    length = length + np.linalg.norm(loc1 - loc2)

                children = tree.successors(node)
                for child in children:
                    if child != path[i + 1]:
                        other_trees.append(
                            (
                                nx.algorithms.traversal.depth_first_search.dfs_tree(
                                    tree, source=child
                                ),
                                node,
                                length,
                            )
                        )

        path = list(path)

        return (path, other_trees)

    def path_length(self, path):

        length = 0

        for i, node in enumerate(path):
            if i > 0:
                length = length + np.linalg.norm(
                    self.nodes[node]["loc"] - self.nodes[path[i - 1]]["loc"]
                )
        return length

    def fit_spline_tree_branch(self):
        spline_tree = nx.DiGraph()
        curr_spline_num = 0
        stack = []
        root = self.root
        successors = nx.algorithms.traversal.depth_first_search.dfs_successors(
            self, source=root
        )

        path, length = self.get_segment_branch(successors, root)
        spline_tree.add_node(curr_spline_num, path=path, starting_length=0)
        end_pt = path[-1]

        if end_pt in successors.keys():
            children = successors[end_pt]

            for child in reversed(children):
                item = (child, curr_spline_num, length)
                stack.append(item)

        while len(stack) > 0:
            curr_spline_num = curr_spline_num + 1
            item = stack.pop()
            start = item[0]
            parent_num = item[1]
            start_len = item[2]
            branch_pt = spline_tree.nodes[parent_num]["path"][-1]
            path, length = self.get_segment_branch(successors, start)
            path.insert(0, branch_pt)
            end_pt = path[-1]

            spline_tree.add_node(curr_spline_num, path=path, starting_length=start_len)
            spline_tree.add_edge(parent_num, curr_spline_num)

            if end_pt in successors.keys():
                children = successors[end_pt]

                for child in reversed(children):
                    item = (child, curr_spline_num, start_len + length)
                    stack.append(item)

        for node in spline_tree.nodes:
            path = spline_tree.nodes[node]["path"]
            spline_tree.nodes[node]["spline"] = self.fit_spline_path(path)

        return spline_tree

    def get_segment_branch(self, successors, start):
        child = start
        length = 0
        sequence = [child]
        try:
            children = successors[child]

            while len(children) == 1:
                child = children[0]
                sequence.append(child)
                length = length + np.linalg.norm(
                    self.nodes[sequence[-1]]["loc"] - self.nodes[sequence[-2]]["loc"]
                )
                children = successors[child]
        except KeyError:
            pass
        return sequence, length

    def fit_spline_tree_x(self):
        # depth first search to get segments
        # constraints - equals first value, c3 through a segment
        # min function error
        splines = []  # each element is (tstart,tend,5-array of coefficients)
        stack = []
        root = 1
        dfs = nx.algorithms.traversal.depth_first_search.dfs_successors(
            self, source=root
        )
        cur_node = root
        cur_len = 0
        cur_children = dfs[cur_node]
        cur_loc = self.nodes[cur_node]["loc"]

        cur_lens = [cur_len]
        cur_xs = [cur_loc]

        lens = [0]
        xs = [cur_loc]

        A = None
        f = None

        while (len(cur_children) > 0) or (len(stack) > 0):
            if len(cur_children) > 0:
                for c, child in enumerate(cur_children):
                    if c == 0:
                        cur_child = child
                    else:
                        stack.append((cur_len, cur_node, child))
                cur_node = cur_child
                cur_len = cur_len + np.abs(cur_loc[0] - self.nodes[cur_node]["loc"][0])
                cur_loc = self.nodes[cur_node]["loc"]

                cur_lens.append(cur_len)
                cur_xs.append(cur_loc)

                lens.append(cur_len)
                xs.append(cur_loc)
            else:
                parent_len, parent, cur_node = stack.pop()
                cur_loc = self.nodes[cur_node]["loc"]
                cur_len = parent_len + np.abs(cur_loc[0] - self.nodes[parent]["loc"][0])

                cur_lens = [parent_len, cur_len]
                cur_xs = [self.nodes[parent]["loc"], cur_loc]

                lens.append(cur_len)
                xs.append(cur_loc)

            try:
                cur_children = dfs[cur_node]
            except KeyError:
                cur_children = []
                x = np.stack(cur_xs)
                m = x.shape[0]
                tck, u = splprep([x[:, 0], x[:, 1], x[:, 2]], k=np.amin([m - 1, 5]))
                splines.append((u, tck))

                """
                n = 5 * (len(cur_lens)-1)
                p = len(cur_lens) - 1
                m = (len(cur_lens) - 2) * 4 + 1
                A = np.zeros([p, n])
                f = np.zeros([p])
                CT = np.zeros([m,n])
                b = np.zeros([m])

                b[0] = cur_xs[0]
                CT[0, :5] = [1, cur_lens[0], cur_lens[0]**2, cur_lens[0]**3, cur_lens[0]**4]

                for i in np.arange(1, len(cur_lens)):
                    col = 5*(i-1)

                    A[i-1, col:col+5] = [1, cur_lens[i], cur_lens[i]**2, cur_lens[i]**3, cur_lens[i]**4]
                    f[i-1] = cur_xs[i]

                    if i <  len(cur_lens) - 1:
                        row = 4*(i-1) + 1

                        CT[row, col:col+5] = [1, cur_lens[i], cur_lens[i]**2, cur_lens[i]**3, cur_lens[i]**4]
                        CT[row, col+5:col+10] = [-1, -cur_lens[i], -cur_lens[i]**2, -cur_lens[i]**3, -cur_lens[i]**4]

                        CT[row+1, col+1:col+5] = [1, 2*cur_lens[i], 3*cur_lens[i]**2, 4*cur_lens[i]**3]
                        CT[row+1, col+6:col+10] = [-1, -2*cur_lens[i], -3*cur_lens[i]**2, -4*cur_lens[i]**3]

                        CT[row+2, col+2:col+5] = [2, 6*cur_lens[i], 12*cur_lens[i]**2]
                        CT[row+2, col+7:col+10] = [-2, -cur_lens[i], -12*cur_lens[i]**2]

                        CT[row+3, col+3:col+5] = [6, 24*cur_lens[i]]
                        CT[row+3, col+8:col+10] = [-6, -24*cur_lens[i]]

                G = np.eye(n)*0.01 + A.T@A
                a = A.T @ f
                C = CT.T

                c = quadprog.solve_qp(G, a, C, b, meq=len(b))[0]

                for i in np.arange(0, len(cur_lens)-1):
                    coeffs = c[5*i:5*(i+1)]
                    tstart = cur_lens[i]
                    tend = cur_lens[i+1]

                    quartics.append((tstart, tend, coeffs))
                """

        return lens, xs, splines
