import ngauge
import numpy as np
import networkx as nx
from brainlit.algorithms.trace_analysis.fit_spline import CubicHermiteChain
from scipy.interpolate import splprep, splev
from tqdm import tqdm
from similaritymeasures import frechet_dist


def replace_root(neuron):
    """nGauge neurons have a distinct branch for every dendrite/axon that emanates from the soma. Since we only work with axon traces, this function combines all branches into one.

    Args:
        neuron (ngauge.Neuron): nGaguge neuron whose branches will be combined.

    Returns:
        ngauge.Neuron: nGaguge neuron with a single branch.
    """
    # assert all branch heads have the same data
    for branch_n, branch_head in enumerate(neuron.branches):
        if branch_n == 0:
            first_data = np.array(
                [branch_head.x, branch_head.y, branch_head.z, branch_head.r]
            )
        else:
            data = np.array(
                [branch_head.x, branch_head.y, branch_head.z, branch_head.r]
            )
            assert np.array_equal(first_data, data)

    # make a single branch head
    first_root = neuron.branches[0]
    for i in range(1, len(neuron.branches)):
        first_root.children += neuron.branches[i].children

    # change parents of children
    for child in first_root.children:
        child.parent = first_root

    neuron.branches = [first_root]
    neuron.soma_layers = {}
    return neuron


def split_paths(node):
    """Split the subtree under an nGauge trace point into paths by recursively identifying the longest root to leaf path.

    Args:
        node (ngauge.TracingPoint): nGauge trace point that defines the subtree that will be split.

    Returns:
        list: set of paths that compose the subtree.
    """
    stack = [(node, None)]

    paths = []
    while len(stack) > 0:
        root = stack.pop()
        _, longest_path = find_longest_path(node=root[0])
        stack += remove_path(longest_path)

        if root[1] != None:
            longest_path = [root[1]] + longest_path

        paths.append(longest_path)

    return paths


def find_longest_path(node):
    """Find longest path that reaches a leaf.

    Args:
        node (ngauge.TracingPoint): nGauge trace point from which this method will look for descending paths.

    Returns:
        float: Path length of longest node to leaf path.
        list: Longest node to leaf path.
    """
    if len(node.children) == 0:
        return 0, [node]
    else:
        longest_dist = 0
        longest_path = []
        for child in node.children:
            dist, path = find_longest_path(child)
            new_dist = dist + np.linalg.norm(
                np.subtract([node.x, node.y, node.z], [child.x, child.y, child.z])
            )
            if new_dist > longest_dist:
                longest_path = [node] + path
                longest_dist = new_dist

        return longest_dist, longest_path


def remove_path(path):
    """Remove a path from a tree, by collecting all remaining subtrees in a list.

    Args:
        path (list): List of nGauge TracingPoints which identifies the path to be removed. Built to work with find_longest_path within the split_paths method.

    Raises:
        ValueError: Two consecutive nodes in the path do not have a parent-child connection.

    Returns:
        list: List of remaining subtrees (ngauge TracingPoints).
    """
    subtrees = []
    for p, p_next in zip(path[:-1], path[1:]):
        if len(p.children) > 1:
            child_found = False
            for child in p.children:
                if child != p_next:
                    subtrees.append((child, p))
                else:
                    child_found = True
            if not child_found:
                raise ValueError(f"{p_next} not found as child of {p}")

    return subtrees


class ZerothFirstOrderNeuron:
    """Class used to combine a Diffeomorphic Action with an nGauge Neuron."""

    def __init__(self, neuron, da=None, sampling=None):
        """Apply a diffeomorphis action to an nGauge Neuron trace.

        Args:
            neuron (ngauge.Neuron): Neuron trace.
            da (DiffeomorphismAction, optional): Action to be applied to the neuron trace. Defaults to None.
            sampling (float, optional): Sampling distance of the ground truth and discrete mappings, in microns. Defaults to None.
        """
        self.sampling = sampling

        neuron = replace_root(neuron)
        DG = self._create_path_graph(neuron)

        if da is not None:
            DG = self._ground_truth(DG, da)
            DG = self._zeroth_order(DG, da)
            DG = self._first_order(DG, da)

        self.DG = DG

    def _create_path_graph(self, neuron):
        paths = split_paths(neuron.branches[0])  # assumes first path is the longest
        root = paths[0][0]

        DG = nx.DiGraph()
        for p_idx, path in enumerate(paths):
            coords = []
            for node in path:
                coords.append([node.x, node.y, node.z])

            DG.add_node(p_idx, coords=np.array(coords), path=path)

        for path_node1 in DG.nodes:
            # skip longest path because it will be root
            if path_node1 == 0:
                continue

            path_root = DG.nodes[path_node1]["path"][0]

            # address all other paths that branch from root point
            if path_root == root:
                DG.add_edge(0, path_node1, connect_point=0)
                continue

            for path_node2 in DG.nodes:
                path2 = DG.nodes[path_node2]["path"]
                for p_idx, pt_node in enumerate(path2):
                    if pt_node == path_root and p_idx > 0:
                        DG.add_edge(path_node2, path_node1)

        assert nx.is_weakly_connected(DG)
        assert nx.number_of_nodes(DG) == nx.number_of_edges(DG) + 1
        return DG

    def _ground_truth(self, DG, da):
        sampling = self.sampling
        for path_node in DG.nodes:
            coords = DG.nodes[path_node]["coords"]
            us = [0] + list(np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
            DG.nodes[path_node]["u"] = us

            tck, us = splprep(
                [coords[:, i] for i in range(coords.shape[1])], k=1, s=0, u=us
            )

            us_us = [us[0]]
            for u1, u2 in zip(us[:-1], us[1:]):
                if sampling is not None and u2 - u1 > sampling:
                    us_us += list(np.arange(u1 + sampling, u2, sampling))
                us_us += [u2]

            coords_us = splev(us_us, tck)
            coords_us = np.stack(coords_us, axis=1)
            if sampling is None:
                assert np.allclose(coords_us, coords)

            new_coords = da.evaluate(coords_us)
            spline = splprep(
                [new_coords[:, i] for i in range(new_coords.shape[1])],
                k=1,
                u=us_us,
                s=0,
            )

            DG.nodes[path_node]["gt"] = spline
        return DG

    def _zeroth_order(self, DG, da):
        for path_node in DG.nodes:
            coords = DG.nodes[path_node]["coords"]
            us = DG.nodes[path_node]["u"]

            new_coords = da.evaluate(coords)
            spline = splprep(
                [new_coords[:, i] for i in range(new_coords.shape[1])], k=1, u=us, s=0
            )

            DG.nodes[path_node]["spline_0"] = spline

        return DG

    def _first_order(self, DG, da):
        for path_node in DG.nodes:
            x = DG.nodes[path_node]["u"]
            y = DG.nodes[path_node]["coords"]
            diffs = np.diff(y, axis=0)
            diffs /= np.repeat(np.linalg.norm(diffs, axis=1)[:, np.newaxis], 3, axis=1)
            left_dydx = diffs
            right_dydx = diffs

            new_left_dydx = da.D(y[:-1, :], left_dydx)
            new_right_dydx = da.D(y[1:, :], right_dydx)
            new_y = da.evaluate(y)

            spline = CubicHermiteChain(
                x=x, y=new_y, left_dydx=new_left_dydx, right_dydx=new_right_dydx
            )

            DG.nodes[path_node]["spline_1"] = spline

        return DG

    def _make_path_pts(self, path_node, root=False):
        DG = self.DG
        sampling = self.sampling

        us = DG.nodes[path_node]["u"]
        tck, _ = DG.nodes[path_node]["spline_0"]

        if root:
            t1 = 1
        else:
            t1 = 2

        pt = splev(us[0], tck)
        pt = [float(p) for p in pt]
        cur_pt_0 = ngauge.TracingPoint(x=pt[0], y=pt[1], z=pt[2], r=1, t=t1)
        root_0 = cur_pt_0

        spline1 = DG.nodes[path_node]["spline_1"]
        pt = spline1(us[0])
        pt = [float(p) for p in pt]
        cur_pt_1 = ngauge.TracingPoint(x=pt[0], y=pt[1], z=pt[2], r=1, t=t1)
        root_1 = cur_pt_1

        for u1, u2 in zip(us[:-1], us[1:]):
            if sampling is not None and u2 - u1 > sampling:
                us_inter = np.arange(u1 + sampling, u2, sampling)

                for u_inter in us_inter:
                    pt = splev(u_inter, tck)
                    pt = [float(p) for p in pt]
                    new_pt_0 = ngauge.TracingPoint(
                        x=pt[0], y=pt[1], z=pt[2], r=1, t=2, parent=cur_pt_0
                    )
                    cur_pt_0.children = [new_pt_0]
                    cur_pt_0 = new_pt_0

                    pt = spline1(u_inter)
                    pt = [float(p) for p in pt]
                    new_pt_1 = ngauge.TracingPoint(
                        x=pt[0], y=pt[1], z=pt[2], r=1, t=2, parent=cur_pt_1
                    )
                    cur_pt_1.children = [new_pt_1]
                    cur_pt_1 = new_pt_1

            pt = splev(u2, tck)
            pt = [float(p) for p in pt]
            new_pt_0 = ngauge.TracingPoint(
                x=pt[0], y=pt[1], z=pt[2], r=1, t=2, parent=cur_pt_0
            )
            cur_pt_0.children = [new_pt_0]
            cur_pt_0 = new_pt_0

            pt = spline1(u2)
            pt = [float(p) for p in pt]
            new_pt_1 = ngauge.TracingPoint(
                x=pt[0], y=pt[1], z=pt[2], r=1, t=2, parent=cur_pt_1
            )
            cur_pt_1.children = [new_pt_1]
            cur_pt_1 = new_pt_1

        return root_0, root_1

    def frechet_errors_path(self, path_node):
        DG = self.DG
        root_0, root_1 = self._make_path_pts(path_node)
        tck, u = DG.nodes[path_node]["gt"]
        coords = splev(u, tck)
        gt_coords = np.stack(coords, axis=1)

        zero_coords = []
        stack = []
        stack += root_0
        while stack:
            child = stack.pop()
            stack += child.children
            zero_coords.append([child.x, child.y, child.z])

        zero_coords = np.array(zero_coords)

        first_coords = []
        stack = []
        stack += root_1
        while stack:
            child = stack.pop()
            stack += child.children
            first_coords.append([child.x, child.y, child.z])

        first_coords = np.array(first_coords)

        zero_error = frechet_dist(gt_coords, zero_coords)
        first_error = frechet_dist(gt_coords, first_coords)

        return zero_error, first_error

    def frechet_errors(self):
        """Compute frechet errors between ground truth and both zeroth and first order mapping.

        Returns:
            float: Frechet error between zeroth order mapping and ground truth.
            float: Frechet error between first order mapping and ground truth.
        """
        DG = self.DG
        max_zero_error = 0
        max_first_error = 0
        for path_node in tqdm(
            DG.nodes, desc="iterating through branches...", leave=False
        ):
            zero_error, first_error = self.frechet_errors_path(path_node)

            if zero_error > max_zero_error:
                max_zero_error = zero_error
            if first_error > max_first_error:
                max_first_error = first_error

        return max_zero_error, max_first_error

    def get_transforms(self):
        """Get zeroth and first order mapped neurons in ngauge form. Action is done on each branch, then branches are reattached by looking at coordinate positions.

        Raises:
            ValueError: Some branches were not reattached.
            ValueError: Some branches were not reattached.

        Returns:
            ngauge.Neuron: Transformed neuron trace via zeroth order mapping.
            ngauge.Neuron: Transformed neuron trace via first order mapping.
        """
        DG = self.DG

        proots_0 = []
        proots_1 = []
        for p_idx, path_node in enumerate(DG.nodes):
            if p_idx == 0:  # assumes first path has root
                root_0, root_1 = self._make_path_pts(path_node, root=True)
            else:  # find path and add to tree
                proot_0, proot_1 = self._make_path_pts(path_node)
                proots_0.append(proot_0)
                proots_1.append(proot_1)

        with tqdm(total=len(proots_0), desc="joining branches 0", leave=False) as pbar:
            while len(proots_0) > 0:
                start_len = len(proots_0)
                for pr_idx, proot in enumerate(proots_0):
                    proot_loc = np.array([proot.x, proot.y, proot.z])
                    stack = []
                    stack += [root_0]
                    while len(stack) > 0:
                        node = stack.pop()
                        stack += node.children

                        node_loc = np.array([node.x, node.y, node.z])
                        if np.linalg.norm(proot_loc - node_loc) < 1e-6:
                            attach_pt = proot.children[0]
                            attach_pt.parent = node
                            node.children += [attach_pt]
                            proots_0.pop(pr_idx)
                            break
                if len(proots_0) == start_len:
                    raise ValueError(f"No branches were joined")
                else:
                    pbar.update(start_len - len(proots_0))

        neuron_0 = ngauge.Neuron()
        neuron_0.add_branch(root_0)

        with tqdm(total=len(proots_1), desc="joining branches 1", leave=False) as pbar:
            while len(proots_1) > 0:
                start_len = len(proots_1)
                for pr_idx, proot in enumerate(proots_1):
                    proot_loc = np.array([proot.x, proot.y, proot.z])
                    stack = []
                    stack += [root_1]
                    while len(stack) > 0:
                        node = stack.pop()
                        stack += node.children

                        node_loc = np.array([node.x, node.y, node.z])
                        if np.linalg.norm(proot_loc - node_loc) < 1e-6:
                            attach_pt = proot.children[0]
                            attach_pt.parent = node
                            node.children += [attach_pt]
                            proots_1.pop(pr_idx)
                            break
                if len(proots_1) == start_len:
                    raise ValueError(f"No branches were joined")
                else:
                    pbar.update(start_len - len(proots_1))

        neuron_1 = ngauge.Neuron()
        neuron_1.add_branch(root_1)

        return neuron_0, neuron_1

    def _make_paths_pts_gt(self, path_node, root=False):
        DG = self.DG

        tck, u = DG.nodes[path_node]["gt"]
        pts = splev(u, tck)
        pts = np.stack(pts, axis=1)
        pt = pts[0, :]
        pt = [float(p) for p in pt]

        if root:
            t1 = 1
        else:
            t1 = 2

        cur_pt = ngauge.TracingPoint(x=pt[0], y=pt[1], z=pt[2], r=1, t=t1)
        root = cur_pt

        for pt in pts[1:, :]:
            pt = [float(p) for p in pt]
            new_pt = ngauge.TracingPoint(
                x=pt[0], y=pt[1], z=pt[2], r=1, t=2, parent=cur_pt
            )
            cur_pt.children = [new_pt]
            cur_pt = new_pt

        return root

    def get_gt(self):
        """Get the ground truth mapping in ngauge Neuron form. Ground truth mapping is defined as upsampling the original trace (linear interpolation), the zeroth order mapping of these points.

        Raises:
            ValueError: Some branches were not reattached.

        Returns:
            ngauge.Neuron: Ground truth mapping.
        """
        DG = self.DG

        proots = []
        for p_idx, path_node in enumerate(DG.nodes):
            if p_idx == 0:  # assumes first path has root
                root = self._make_paths_pts_gt(path_node, root=True)
            else:  # find path and add to tree
                proot = self._make_paths_pts_gt(path_node)
                proots.append(proot)

        with tqdm(total=len(proots), desc="joining branches gt", leave=False) as pbar:
            while len(proots) > 0:
                start_len = len(proots)
                for pr_idx, proot in enumerate(proots):
                    proot_loc = np.array([proot.x, proot.y, proot.z])
                    stack = []
                    stack += [root]
                    while len(stack) > 0:
                        node = stack.pop()
                        stack += node.children

                        node_loc = np.array([node.x, node.y, node.z])
                        if np.linalg.norm(proot_loc - node_loc) < 1e-6:
                            attach_pt = proot.children[0]
                            attach_pt.parent = node
                            node.children += [attach_pt]
                            proots.pop(pr_idx)
                            break
                if len(proots) == start_len:
                    raise ValueError(f"No branches were joined")
                else:
                    pbar.update(start_len - len(proots))

        neuron = ngauge.Neuron()
        neuron.add_branch(root)

        return neuron
