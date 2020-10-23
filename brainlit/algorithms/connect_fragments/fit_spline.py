#%%
import numpy as np
from skimage.measure import label
from skimage.morphology import skeletonize
from sklearn.feature_extraction.image import grid_to_graph
import scipy.ndimage as ndi
from scipy import optimize
from scipy.interpolate import splrep, splprep, interp1d
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

"""
Function definitions
"""

def checkIfDuplicates_2(listOfElems):
    """Check if given list contains any duplicates

    """   
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
    """[summary]

    Args:
        nx ([type]): [description]
    """
    def __init__(self):
        super(GeometricGraph, self).__init__()
        self.segments = None
        self.cycle = None
        self.root = 1

    def set_root(self, root):
        self.root_node = root

    def check_closed(self,node,edges):
        G=GeometricGraph()
        G.edges.data()


    # main function
    
    def fit_spline_tree_invariant(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        '''
        Parameters:
        1. spline_tree: a geometric graph
        2. curr_spline_num: current spline number, used to ??
        3. stack: list of 
        4. root: the only node without parent; defined as 1; not to confused with 'roots'
        5. tree: an oriented tree w.r.t. root
        6. path: a list of nodes
        7. other_trees: 
        8. starting_length: 
        9. treenum: 
        10. parent_num: 
        '''
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
        """[summary]

        Args:
            path ([type]): [description]

        Returns:
            [type]: [description]
        """
        '''

        Parameters:
        1. x:
        2. orig:
        3. new:
        4. m:
        5. diffs:
        6. k:
        7. tck:
        8. u:
        '''
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

    def check_multiplicity(self, t):
        """[summary]

        Args:
            t ([type]): [description]

        Raises:
            RuntimeError: [description]
        """
        '''
        check multiplicity

        Parameters:
        1. knots:
        2. t:
        3. first: first element of knots
        4. indices_keep:
        5. dup: 
        '''
        knots = list(t.copy())
        first = knots[0]
        last = knots[-1]
        indices_keep = (knots != first) & (knots != last)
        knots = [i for (i,v) in zip(knots, indices_keep) if v]
        dup = checkIfDuplicates_2(knots)
        if dup:
            print(t)
            raise RuntimeError("Duplicates found in the above knot list")
    
    
    # find the longest path
    def find_longest_path(self, tree, starting_length=0):
        """[summary]

        Args:
            tree ([type]): [description]
            starting_length (int, optional): [description]. Defaults to 0.

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        '''
        Return:
        1. path
        2. other_trees

        Parameters:
        1. other_trees: 
        2. roots: a list of nodes with in-degree=0, used to make sure there exist only one root
        3. leaves: nodes without children
        4. shortest_paths: selected paths in leaves that are shortest in length
        5. distances: the lengths of each path in shortest_paths
        6. shortest_idx: the maximal value in distances
        7. leaf: the shortest element of leaves
        8. length: the length between nodes
        9. children: the seccessors (or nodes) followed by a specific node
        10. child: elements in children
        11. path: a list of nodes

        Modules:
        in_degree: number of edges pointing to the node
        out_degree: number of edges pointing out of the node

        '''
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
        """[summary]

        Args:
            path ([type]): [description]

        Returns:
            [type]: [description]
        """
        '''
        compute the distance between nodes along the path
        '''
        length = 0

        for i, node in enumerate(path):
            if i > 0:
                length = length + np.linalg.norm(
                    self.nodes[node]["loc"] - self.nodes[path[i - 1]]["loc"]
                )
        return length

# %%
