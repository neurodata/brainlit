import pytest
import networkx as nx

def test_fit_spline_tree_invariant(self):
        spline_tree = nx.DiGraph()
        curr_spline_num = 0
        assert curr_spline_num ==0
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