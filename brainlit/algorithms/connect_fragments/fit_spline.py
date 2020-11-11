import numpy as np
from scipy.interpolate import splprep
import math
import warnings
import networkx as nx
import itertools
from brainlit.utils.util import (
    check_type,
    check_size,
    check_precomputed,
    check_iterable_type,
    check_iterable_nonnegative,
)


"""
Geometric Graph class
"""


class GeometricGraph(nx.Graph):
    """class for undirected graphs
    Args:
        nx (Graph): A Graph stores nodes and edges with optional data, or attributes.
    """

    def __init__(self):
        super(GeometricGraph, self).__init__()
        self.segments = None
        self.cycle = None
        self.root = 1

    def fit_spline_tree_invariant(self):
        """construct a spline tree based on the path lengths
        Raises:
            ValueError: check if every node is unigue in location
            ValueError: check if every node is assigned to at least one edge
            ValueError: check if the graph contains undirected cycle(s)
            ValueErorr: check if the graph has disconnected segment(s)
        Returns:
            spline_tree (DiGraph): a parent tree with the longest path in the directed graph
        """

        # check integrity of 'loc' attributes in the neuron
        if any([self.nodes[node].get("loc") is None for node in self.nodes]):
            raise KeyError("some nodes are missing the 'loc' attribute")
        for node in self.nodes:
            check_type(self.nodes[node].get("loc"), np.ndarray)
        if any([self.nodes[node].get("loc").ndim != 1 for node in self.nodes]):
            raise ValueError("nodes must be flat arrays")
        if any([len(self.nodes[node].get("loc")) == 0 for node in self.nodes]):
            raise ValueError("nodes cannot have empty 'loc' attributes")
        for node in self.nodes:
            check_iterable_type(self.nodes[node].get("loc"), (np.integer, np.float))
        if any([len(self.nodes[node].get("loc")) != 3 for node in self.nodes]):
            raise ValueError("'loc' attributes must contain 3 coordinates")

        # check there are no duplicate nodes
        LOCs = [np.ndarray.tolist(self.nodes[node]["loc"]) for node in self.nodes]
        LOCs.sort()
        unique_LOCs = list(LOC for LOC, _ in itertools.groupby(LOCs))
        if len(LOCs) != len(unique_LOCs):
            raise ValueError("there are duplicate nodes")

        # check the graph is edge-covering
        if not nx.algorithms.is_edge_cover(self, self.edges):
            raise ValueError("the edges are not a valid cover of the graph")
        # check there are no undirected cycles in the graph
        if not nx.algorithms.tree.recognition.is_forest(self):
            raise ValueError("the graph contains undirected cycles")
        # check there are no disconnected segments
        if not nx.algorithms.tree.recognition.is_tree(self):
            raise ValueError("the graph contains disconnected segments")

        spline_tree = nx.DiGraph()
        curr_spline_num = 0
        stack = []
        root = self.root
        tree = nx.algorithms.traversal.depth_first_search.dfs_tree(self, source=root)
        main_branch, collateral_branches = self.__find_main_branch(tree)
        spline_tree.add_node(curr_spline_num, path=main_branch, starting_length=0)

        for tree in collateral_branches:
            stack.append((tree, curr_spline_num))

        while len(stack) > 0:
            curr_spline_num = curr_spline_num + 1
            treenum = stack.pop()
            tree = treenum[0]
            parent_num = treenum[1]

            main_branch, collateral_branches = self.__find_main_branch(
                tree[0], starting_length=tree[2]
            )
            main_branch.insert(0, tree[1])

            spline_tree.add_node(
                curr_spline_num, path=main_branch, starting_length=tree[2]
            )
            spline_tree.add_edge(parent_num, curr_spline_num)

            for tree in collateral_branches:
                stack.append((tree, curr_spline_num))

        for node in spline_tree.nodes:
            main_branch = spline_tree.nodes[node]["path"]

            spline_tree.nodes[node]["spline"] = self.__fit_spline_path(main_branch)

        return spline_tree

    def __fit_spline_path(self, path):
        """calculate the knots, B-spline coefficients, and the degree of the spline according to the path
        Args:
            path (list): a list of nodes
        Raises:
            ValueError: Nodes should be defined under loc attribute
            TypeError: loc should be of numpy.ndarray class
            ValueError: loc should be 3-dimensional
        Returns:
            tck (tuple): (t,c,k) a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline.
            u (): An array of the values of the parameter.
        """

        # check if loc is defined and of numpy.ndarray class
        for row, node in enumerate(path):
            hasloc = self.nodes[node].get("loc")
            if hasloc is None:
                raise ValueError("Nodes are not defined under loc attribute")
            elif type(hasloc) is not np.ndarray:
                raise TypeError("loc is not a numpy.ndarray")
            elif len(hasloc) != 3:
                raise ValueError("loc is not 3-dimensional")

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

        self.__check_multiplicity(tck[0])

        return tck, u


    def __find_main_branch(self, tree: nx.DiGraph, starting_length: float = 0):
        r"""Find the main branch in a directed graph.
        It is used in `fit_spline_tree_invariant` to identify the main branch
        in a neuron and group the collateral branches for later analysis.
        The main branch is defined as the longest possible path connecting the
        neuron's nodes, in terms of spatial distance. An example is provided in
        the following figure:
        .. figure:: https://raw.githubusercontent.com/neurodata/brainlit/develop/docs/images/find_main_branch.png
            :scale: 25%
            :alt: find_main_branch example
            Graphic example of `find_main_branch()` functionality.
        Arguments:
            tree: nx.DiGraph, a directed graph.
                It is the result of nx.algorithms.traversal.depth_first_search.dfs_tree()
                which returns an oriented tree constructed from a depth-first search of
                the neuron.
            starting_length: float, optional.
                It is the spatial distance between the root of the neuron (i.e `self.root`) and
                the root of the current main branch. It must be real-valued, non-negative.
                It is defaulted to `0` for the first main branch, that starts from the root of
                the neuron.
        Returns:
            main_branch: list, a list of nodes.
            collateral_branches: list, directed graphs of children trees.
        """

        # Initialize the list of collateral branches
        collateral_branches = []
        # If there is only one node in the tree, that is the main branch
        if len(tree.nodes) == 1:
            main_branch = tree.nodes
        else:
            # Find the root of the tree.
            # A node is a candidate to be the root if it does not
            # have any edges pointing to it (i.e. in_degree == 0)
            roots = [node for node, degree in tree.in_degree() if degree == 0]
            root = roots[0]
            # Find the leaves of the tree.
            # A node is a leaf if it has only one edge pointing
            # to it (i.e. in_degree == 1), and no edges pointing
            # out of it (i.e. out_degree == 0)
            leaves = [
                node
                for node in tree.nodes()
                if tree.out_degree(node) == 0 and tree.in_degree(node) == 1
            ]
            # For each leaf, compute the shortest path to reach it
            shortest_paths = [
                nx.algorithms.shortest_paths.generic.shortest_path(
                    tree, source=root, target=l
                )
                for l in leaves
            ]
            # Compute the lengths of the paths
            lengths = [self.__path_length(path) for path in shortest_paths]
            # Find the longest path
            longest_path_idx = np.argmax(lengths)
            furthest_leaf = leaves[longest_path_idx]
            # Find the main branch
            main_branch = nx.algorithms.shortest_paths.generic.shortest_path(
                tree, source=root, target=furthest_leaf
            )

            # Here, we walk on the main branch to find
            # the collateral branches
            for i, node in enumerate(main_branch):
                # Increase starting_length by the size of
                # the step on the main branch
                if i > 0:
                    loc1 = self.nodes[node]["loc"]
                    loc2 = self.nodes[main_branch[i - 1]]["loc"]
                    starting_length += np.linalg.norm(loc2 - loc1)
                # Find all successors of the current node on
                # the main branch. A node m is a successor of the node
                # n if there is a directed edge that goes from n to m
                children = tree.successors(node)
                for child in children:
                    # If the successor is not on the main branch, then
                    # we found a branching point of the neuron
                    if child != main_branch[i + 1]:
                        # Explore the newly-found branch and
                        # append it to the list of collateral branches
                        collateral_branches.append(
                            (
                                nx.algorithms.traversal.depth_first_search.dfs_tree(
                                    tree, source=child
                                ),
                                node,
                                starting_length,
                            )
                        )
        return list(main_branch), collateral_branches

    def __path_length(self, path):
        r"""Compute the length of a path.
        Given a path ::math::`p = (r_1, \dots, r_N)`, where
        ::math::`r_k = [x_k, y_k, z_k], k = 1, \dots, N`, the length
        `l` of a path is computed as the sum of the lengths of the
        edges of the path. We can write:
        .. math::
            l = \sum_{k=2}^N \lVert r_k - r_{k-1} \rVert
        Arguments:
            path: a list of nodes.
                The integrity of the nodes is checked for at the beginning of
                `fit_spline_tree_invariant`.
        Returns:
            length: float.
                It is the length of the path.
        """

        length = sum(
            [
                np.linalg.norm(self.nodes[node]["loc"] - self.nodes[path[i - 1]]["loc"])
                if i >= 1
                else 0
                for i, node in enumerate(path)
            ]
        )
        return length
