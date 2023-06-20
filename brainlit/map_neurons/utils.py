import ngauge
import numpy as np
import networkx as nx
from brainlit.algorithms.trace_analysis.fit_spline import CubicHermiteChain
from scipy.interpolate import splprep, splev
from tqdm import tqdm

def replace_root(neuron):

    # assert all branch heads have the same data
    for branch_n, branch_head in enumerate(neuron.branches):
        if branch_n == 0:
            first_data = np.array([branch_head.x,branch_head.y,branch_head.z,branch_head.r])
        else:
            data = np.array([branch_head.x,branch_head.y,branch_head.z,branch_head.r])
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

def resample_neuron(neuron, sampling):
    neuron = replace_root(neuron)
    
    stack = []
    stack += neuron.branches[0].children

    while len(stack) > 0:
        child = stack.pop()
        stack += child.children

        parent = child.parent
        parents_children = parent.children
        for idx, c in enumerate(parents_children):
            if c == child:
                child_idx = idx
                break

        pt1 = np.array([parent.x, parent.y, parent.z])
        pt2 = np.array([child.x, child.y, child.z])

        dist = np.linalg.norm(pt2-pt1)

        if dist > sampling:
            samples = np.arange(sampling, dist, sampling)
            
            for n_sample, sample in enumerate(samples):
                loc = (pt2-pt1)/dist*sample+pt1
                loc = [float(l) for l in loc]
                new_pt = ngauge.TracingPoint(x=loc[0], y=loc[1], z=loc[2], r=1, t=child.t)
                if n_sample == 0: #beginning of chain is loose
                    first_pt = new_pt
                else: # add link to chain
                    new_pt.parent = prev_pt
                    prev_pt.children = [new_pt]

                prev_pt = new_pt
            
            # attach end to child
            child.parent = new_pt
            new_pt.children = [child]

            # attach beginning of chain to parent
            first_pt.parent = parent
            parents_children.pop(child_idx)
            parents_children.append(first_pt)

    return neuron

def zeroth_order_map_neuron(neuron, da):
    neuron = replace_root(neuron)

    stack = []
    stack += neuron.branches

    while len(stack) > 0:
        child = stack.pop()
        stack += child.children

        pt = np.array([child.x, child.y, child.z])
        new_loc = da.evaluate(pt)
        child.x, child.y, child.z = new_loc[0,0], new_loc[0,1], new_loc[0,2]

    return neuron

def first_order_map_neuron(neuron, da, sampling=None):
    neuron = replace_root(neuron)

    paths = split_paths(neuron.branches[0])

    DG = nx.DiGraph()
    for p_idx, path in enumerate(paths):
        DG.add_node(p_idx, og_path=path)
    for node in DG.nodes:
        root = DG.nodes[node]["og_path"][0]
        for node2 in DG.nodes:
            if node2 == node:
                continue

            path = DG.nodes[node2]["og_path"]
            for pt in path:
                if root == pt:
                    DG.add_edge(node2, node)

    
    for node in DG.nodes:
        path = DG.nodes[node]["og_path"]
        # find locations and derivatives
        y = [[path[0].x,path[0].y,path[0].z]]
        x = [0]
        left_dydx = []
        right_dydx = []
        for n_idx, node in enumerate(path[1:]):
            new_pt = [node.x,node.y,node.z]
            diff = np.subtract(new_pt, y[n_idx])
            x.append(x[n_idx] + np.linalg.norm(diff))
            y.append(new_pt)
            left_dydx.append(diff/np.linalg.norm(diff))
            right_dydx.append(diff/np.linalg.norm(diff))

        y = np.array(y)
        x = np.array(x)
        left_dydx = np.array(left_dydx)
        right_dydx = np.array(left_dydx)

        # map
        y = da.evaluate(y)
        left_dydx = da.D(y[:-1,:], left_dydx)
        right_dydx = da.D(y[1:,:], right_dydx)

        spline = CubicHermiteChain(x, y, left_dydx, right_dydx)
        path = DG.nodes[node]["transformed_spline"] = spline



def split_paths(node):
    stack = [(node, None)]

    paths = []
    while len(stack) > 0:
        root = stack.pop()
        _, longest_path = find_longest_path(node = root[0])
        stack += remove_path(longest_path)

        if root[1] != None:
            longest_path = [root[1]] + longest_path

        paths.append(longest_path)

    return paths

def find_longest_path(node):
    if len(node.children) == 0:
        return 0, [node]
    else:
        longest_dist = 0
        longest_path = []
        for child in node.children:
            dist, path = find_longest_path(child)
            new_dist = dist + np.linalg.norm(np.subtract([node.x, node.y, node.z], [child.x, child.y, child.z]))
            if new_dist > longest_dist:
                longest_path = [node] + path
                longest_dist = new_dist
        
        return new_dist, longest_path


def remove_path(path):
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
    def __init__(self, neuron, da, sampling):
        self.sampling = sampling

        neuron = replace_root(neuron)
        DG = self.create_path_graph(neuron)

        DG = self.ground_truth(DG, da)
        DG = self.zeroth_order(DG, da)
        DG = self.first_order(DG, da)

        self.DG = DG

    def create_path_graph(self, neuron):
        paths = split_paths(neuron.branches[0]) # assumes first path is the longest
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
        assert nx.number_of_nodes(DG) == nx.number_of_edges(DG)+1
        return DG
    
    def ground_truth(self, DG, da, resample=False):
        sampling = self.sampling
        for path_node in DG.nodes:
            coords = DG.nodes[path_node]["coords"]
            us = [0] + list(np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1)))
            DG.nodes[path_node]["u"] = us

            tck, us = splprep([coords[:,i] for i in range(coords.shape[1])], k=1, s=0, u=us)

            us_us = [us[0]]
            for u1, u2 in zip(us[:-1], us[1:]):
                if resample and u2 - u1 > sampling:
                    us_us += list(np.arange(u1+sampling, u2, sampling))
                us_us += [u2]

            coords_us = splev(us_us, tck)
            coords_us = np.stack(coords_us, axis=1)

            new_coords = da.evaluate(coords_us)
            spline = splprep([new_coords[:,i] for i in range(new_coords.shape[1])], k=1, u=us_us, s=0)

            DG.nodes[path_node]["gt"] = spline
        return DG


    def zeroth_order(self, DG, da):
        for path_node in DG.nodes:
            coords = DG.nodes[path_node]["coords"]
            us = DG.nodes[path_node]["u"]

            new_coords = da.evaluate(coords)
            spline = splprep([new_coords[:,i] for i in range(new_coords.shape[1])], k=1, u=us, s=0)

            DG.nodes[path_node]["spline_0"] = spline

        return DG

    def first_order(self, DG, da):
        for path_node in DG.nodes:
            x = DG.nodes[path_node]["u"]
            y = DG.nodes[path_node]["coords"]
            diffs = np.diff(y, axis=0)
            diffs /= np.repeat(np.linalg.norm(diffs, axis=1)[:, np.newaxis], 3, axis=1)
            left_dydx = diffs
            right_dydx = diffs

            new_left_dydx = da.D(y[:-1,:], left_dydx)
            new_right_dydx = da.D(y[1:,:], right_dydx)
            new_y = da.evaluate(y)

            spline = CubicHermiteChain(x=x, y=new_y, left_dydx=new_left_dydx, right_dydx=new_right_dydx)

            DG.nodes[path_node]["spline_1"] = spline

        return DG
    
    def make_path_pts(self, path_node, resample=False):
        DG = self.DG
        sampling = self.sampling

        us = DG.nodes[path_node]["u"]
        tck, _ = DG.nodes[path_node]["spline_0"]

        pt = splev(us[0], tck)
        pt = [float(p) for p in pt]
        cur_pt_0 = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1)
        root_0 = cur_pt_0

        spline1 = DG.nodes[path_node]["spline_1"]
        pt = spline1(us[0])
        pt = [float(p) for p in pt]
        cur_pt_1 = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1)
        root_1 = cur_pt_1


        for u1, u2 in zip(us[:-1], us[1:]):
            if resample and u2 - u1 > sampling:
                us_inter = np.arange(u1+sampling, u2, sampling)

                for u_inter in us_inter:
                    pt = splev(u_inter, tck)
                    pt = [float(p) for p in pt]
                    new_pt_0 = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1, parent = cur_pt_0)
                    cur_pt_0.children = [new_pt_0]
                    cur_pt_0 = new_pt_0

                    pt = spline1(u_inter)
                    pt = [float(p) for p in pt]
                    new_pt_1 = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1, parent = cur_pt_1)
                    cur_pt_1.children = [new_pt_1]
                    cur_pt_1 = new_pt_1
            

            pt = splev(u2, tck)
            pt = [float(p) for p in pt]
            new_pt_0 = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1, parent = cur_pt_0)
            cur_pt_0.children = [new_pt_0]
            cur_pt_0 = new_pt_0

            pt = spline1(u2)
            pt = [float(p) for p in pt]
            new_pt_1 = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1, parent = cur_pt_1)
            cur_pt_1.children = [new_pt_1]
            cur_pt_1 = new_pt_1

        return root_0, root_1


    def get_transforms(self):
        DG = self.DG

        proots_0 = []
        proots_1 = []
        for p_idx, path_node in enumerate(DG.nodes): 
            if p_idx == 0: # assumes first path has root
                root_0, root_1 = self.make_path_pts(path_node)
            else: #find path and add to tree
                proot_0, proot_1 = self.make_path_pts(path_node)
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
                            proot.children[0].parent = node
                            node.children += [proot]
                            proots_0.pop(pr_idx)
                            break
                if len(proots_0) == start_len:
                    raise ValueError(f"No branches were joined")
                else:
                    pbar.update(start_len-len(proots_0))

        neuron_0 = ngauge.Neuron()
        neuron_0.add_branch(root_0)

        with tqdm(total=len(proots_1), desc="joining branches 1", leave=False) as pbar:
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
                            proot.children[0].parent = node
                            node.children += [proot]
                            proots_0.pop(pr_idx)
                            break
                if len(proots_0) == start_len:
                    raise ValueError(f"No branches were joined")
                else:
                    pbar.update(start_len-len(proots_0))

        neuron_1 = ngauge.Neuron()
        neuron_1.add_branch(root_1)

        return neuron_0, neuron_1

    def make_paths_pts_gt(self, path_node):
        DG = self.DG

        tck, u = DG.nodes[path_node]["gt"]
        pts = splev(u, tck)
        pts = np.stack(pts, axis=1)
        pt = pts[0,:]
        pt = [float(p) for p in pt]
        cur_pt = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1)
        root = cur_pt

        for pt in pts[1:,:]:
            pt = [float(p) for p in pt]
            new_pt = ngauge.TracingPoint(x = pt[0], y=pt[1], z=pt[2], r=1, t=1, parent = cur_pt)
            cur_pt.children = [new_pt]
            cur_pt = new_pt

        return root


    def get_gt(self):
        DG = self.DG

        proots = []
        for p_idx, path_node in enumerate(DG.nodes): 
            if p_idx == 0: # assumes first path has root
                root = self.make_paths_pts_gt(path_node)
            else: #find path and add to tree
                proot = self.make_paths_pts_gt(path_node)
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
                            proot.children[0].parent = node
                            node.children += [proot]
                            proots.pop(pr_idx)
                            break
                if len(proots) == start_len:
                    raise ValueError(f"No branches were joined")
                else:
                    pbar.update(start_len-len(proots))

        neuron = ngauge.Neuron()
        neuron.add_branch(root)

        return neuron 






        

