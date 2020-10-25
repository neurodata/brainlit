#%%
import numpy as np
from scipy.interpolate import splprep
import math
import warnings
import networkx as nx


"""
Function definitions
"""
 
def checkIfDuplicates_2(listOfElems):
    """Check if given list contains any duplicates

    Args:
        listOfElems (set): Build an unordered collection of unique elements.
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
    """class for undirected graphs

    Args:
        nx (Graph): A Graph stores nodes and edges with optional data, or attributes.
    """
    def __init__(self):
        super(GeometricGraph,self).__init__()
        self.segments = None
        self.cycle = None
        self.root = 1


    def fit_spline_tree_invariant(self):
        """construct a spline tree based on the path lengths

        Returns:
            spline_tree (DiGraph): a parent tree with the longest path in the directed graph
        """
        
        spline_tree = nx.DiGraph()
        curr_spline_num = 0
        stack = []
        root = self.root
        tree = nx.algorithms.traversal.depth_first_search.dfs_tree(self, source=root)

        # check if the graph is directed
        if nx.is_directed(tree)== False:
            raise ValueError("The geometric graph is not directed graph")

        # check if the graph is edge covering
        #if nx.is_edge_cover(tree,tree.edges)==False:
        #    raise ValueError("The geometric graph is not edge covering")



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
        """calculate the knots, B-spline coefficients, and the degree of the spline according to the path

        Args:
            path (list): a list of nodes

        Returns:
            tck (tuple): (t,c,k) a tuple containing the vector of knots, the B-spline coefficients, and the degree of the spline.
            u (): An array of the values of the parameter.
        """
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
        """check multiplicity

        Args:
            t (list): the list to be checked

        Raises:
            RuntimeError: when duplicates are found
        """

        knots = list(t.copy())
        first = knots[0]
        last = knots[-1]
        indices_keep = (knots != first) & (knots != last)
        knots = [i for (i,v) in zip(knots, indices_keep) if v]
        dup = checkIfDuplicates_2(knots)
        if dup:
            print(t)
            raise RuntimeError("Duplicates found in the above knot list")
    
    
    def find_longest_path(self, tree, starting_length=0):
        """find the longest path in a tree(Digraph)

        Args:
            tree (Digraph): directed graph
            starting_length (int, optional): Starting length. Defaults to 0.

        Raises:
            ValueError: More than one node with in_degree=0 is prohibited

        Returns:
            path (list): a list of nodes
            other_trees (list): directed graphs of children trees
        """

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
        """compute the distance between nodes along the path

        Args:
            path (list): list of nodes

        Returns:
            length (int): length between nodes
        """
        
        length = 0

        for i, node in enumerate(path):
            if i > 0:
                length = length + np.linalg.norm(
                    self.nodes[node]["loc"] - self.nodes[path[i - 1]]["loc"]
                )
        return length

# %%
