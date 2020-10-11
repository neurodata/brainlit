#%%
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize
from sklearn.feature_extraction.image import grid_to_graph
import scipy.ndimage as ndi
from scipy import optimize
from scipy.interpolate import splrep, splprep, interp1d
from mouselight_code.src import swc2voxel
import cv2
import math
import warnings
import networkx as nx
from joblib import Parallel, delayed
import warnings
from itertools import product
import quadprog
import pickle
import warnings

#keep
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

"""
Function definitions
"""
#keep
def checkIfDuplicates_2(listOfElems):
    ''' Check if given list contains any duplicates '''    
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

    def set_root(self, root):
        self.root_node = root

# main function
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
#keep
    def fit_spline_path(self, path):
        x = np.zeros((len(path), 3))

        for row, node in enumerate(path):
            x[row, :] = self.nodes[node]["loc"]
        orig = x.shape[0]
        x = [xi for i,xi in enumerate(x) if i==0 or (xi!=x[i-1,:]).any()]
        x = np.stack(x, axis=0)
        new = x.shape[0]
        if orig != new:
            warnings.warn(f'{orig-new} duplicate points removed in the trace segment', category=UserWarning)
        m = x.shape[0]
        diffs = np.diff(x, axis=0)
        diffs = np.linalg.norm(diffs, axis=1)
        diffs = np.cumsum(diffs)
        diffs = np.concatenate(([0], diffs))
        k = np.amin([m - 1, 5])
        tck, u = splprep([x[:, 0], x[:, 1], x[:, 2]], u=diffs, k=k)

        self.check_multiplicity(tck[0])

        return tck, u
    # keep
    def check_multiplicity(self, t):
        knots = list(t.copy())
        first = knots[0]
        last = knots[-1]
        indices_keep = (knots != first) & (knots != last)
        knots = [i for (i,v) in zip(knots, indices_keep) if v]
        dup = checkIfDuplicates_2(knots)
        if dup:
            print(t)
            raise RuntimeError("Duplicates found in the above knot list")
    # keep    
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
# keep
    def path_length(self, path):

        length = 0

        for i, node in enumerate(path):
            if i > 0:
                length = length + np.linalg.norm(
                    self.nodes[node]["loc"] - self.nodes[path[i - 1]]["loc"]
                )
        return length
