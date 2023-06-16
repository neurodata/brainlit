from brainlit.map_neurons.diffeo_gen import expR, diffeo_gen_ara
import numpy as np
import matplotlib.pyplot as plt
import torch

from scipy.spatial.distance import cosine
from brainlit.map_neurons.map_neurons import (
    DiffeomorphismAction,
    Diffeomorphism_Transform,
    transform_geometricgraph,
    compute_derivs,
    CloudReg_Transform,
)
import pandas as pd
import seaborn as sns
import os
from cloudvolume import CloudVolume
from pathlib import Path
from brainlit.algorithms.trace_analysis.fit_spline import (
    GeometricGraph,
    compute_parameterization,
    CubicHermiteChain,
)
from brainlit.utils.Neuron_trace import NeuronTrace
from copy import deepcopy
from tqdm import tqdm
from scipy.interpolate import splev, splprep, CubicHermiteSpline
from similaritymeasures import frechet_dist
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
from math import sqrt
import ngauge

# INPUTS
sigmas = [40, 80, 160, 320]
sampling = 2.0
ds_factors = [1, 2, 4, 8, 16]
swc_dir = Path(
    "/cis/home/tathey/projects/mouselight/axon_mapping/ds_experiment/mouselight-swcs"
)
swc_dir = Path(
    "/Users/thomasathey/Documents/mimlab/mouselight/axon_mapping/mouselight-swcs/swcs-1"
)


swcs_fnames = os.listdir(swc_dir)
swcs_fnames = [swc_fname for swc_fname in swcs_fnames if "swc" in swc_fname]
swc_paths = [swc_dir / swc_fname for swc_fname in swcs_fnames]

print(
    f"Processing {len(swc_paths)} neurons with a sampling rate of {sampling} and downsampling factor {ds_factors} with max vs of {sigmas} "
)

def check_duplicates_get_center(ntrace):
    g = ntrace.get_graph()
    coords = []
    for n in g.nodes:
        coords.append([g.nodes[n][i] for i in ["x", "y", "z"]])

    dupes = []
    seen = set()
    for coord in coords:
        coord = tuple(coord)
        if coord in seen:
            dupes.append(coord)
        else:
            seen.add(coord)
    if len(dupes) > 0:
        raise ValueError(f"Duplicate nodes")
    else:
        coords = np.array(coords)
        mx = np.amax(coords, axis=0)
        mn = np.amin(coords, axis=0)
        center = np.mean(np.array([mx, mn]), axis=0)
        return center


def resample_write_swc(neuron, sampling):
    resample_fname = swc_path.stem + f"_resampled{sampling}.swc"
    resample_path = swc_path.parent / resample_fname

    neuron = ngauge.Neuron.from_swc(swc_path)
    neuron = replace_root(neuron)
    
    stack = neuron.branches[0].children

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
                new_pt = ngauge.TracingPoint(x=loc[0], y=loc[1], z=loc[2], r=1, t=child.t)
                if n_sample == 0:
                    first_pt = new_pt
                else:
                    new_pt.parent = prev_pt
                    prev_pt.child = new_pt

                prev_pt = new_pt
            
            child.parent = new_pt
            parents_children.pop(child_idx)
            parents_children.append(first_pt)
            parent.children = parents_children

    return neuron
            






    # ntrace = NeuronTrace(str(swc_path), rounding=False)
    # g = ntrace.get_graph()
    # next_id = np.amax(np.unique(list(g.nodes)))+1

    # edges_to_remove = []
    # for e in g.edges():
    #     pt1 = [g.nodes[e[0]][coord] for coord in ['x','y','z']]

    #     pt2 = [g.nodes[e[1]][coord] for coord in ['x','y','z']]
    #     dist = np.linalg.norm(pt2-pt1)

    #     if dist > sampling:
    #         samples = np.arange(sampling, dist, sampling)
    #         for n_sample, sample in enumerate(samples):
    #             loc = (pt2-pt1)*sample/dist+pt1
    #             g.add_node(next_id, x=loc[0], y=loc[1], z=loc[2])
    #             if n_sample == 0:
    #                 g.add_edge(e[0], next_id)
    #             else:
    #                 g.add_edge(next_id, next_id-1)

    #             if n_sample == len(samples):
    #                 g.add_edge(next_id, e[1])
    #         edges_to_remove.append(e)

    # g.remove_edges_from(edges_to_remove)

    # with open(resample_path, 'wb') as f:








def process_swc(max_r, ct, swc_path):
    ntrace = NeuronTrace(str(swc_path), rounding=False)

    center = check_duplicates_get_center(ntrace=ntrace)

    resample_write_swc(swc_path, sampling=sampling)

    # g = ntrace.get_graph()
    # G_neuron = GeometricGraph()
    # for n in g.nodes:
    #     G_neuron.add_node(
    #         n, loc=np.array([g.nodes[n][i] for i in ["x", "y", "z"]]) - center
    #     )
    # for e in g.edges:
    #     G_neuron.add_edge(e[0], e[1])
    # spline_tree = G_neuron.fit_spline_tree_invariant()

    # for ds_factor in tqdm(
    #     ds_factors, desc="downsample factors...", leave=False, disable=True
    # ):
    #     for branch_id in tqdm(
    #         spline_tree.nodes, desc="Processing branches...", leave=False, disable=True
    #     ):
    #         # Create geometric graph for the branch
    #         path = spline_tree.nodes[branch_id]["path"]
    #         x = []
    #         y = []
    #         z = []
    #         s = []
    #         p = [-1]

    
    #         for point_num, point_id in enumerate(path):
    #             loc = G_neuron.nodes[point_id]["loc"]
    #             x.append(loc[0])
    #             y.append(loc[1])
    #             z.append(loc[2])
    #             s.append(point_num)
    #             if point_num > 0:
    #                 p.append(point_num - 1)

    #         dict = {"x": x, "y": y, "z": z, "sample": s, "parent": p}
    #         df = pd.DataFrame(data=dict)
    #         G_branch = GeometricGraph(df=df, root=s[0])
    #         G_branch.fit_spline_tree_invariant()
    #         spline_tree_branch = G_branch.spline_tree

    #         # downsample
    #         nodes2keep = [node_i for node_i in range(0, len(s), ds_factor)]
    #         if s[-1] not in nodes2keep:
    #             nodes2keep += [s[-1]]
    #         nodes2remove = [node_i for node_i in s if node_i not in nodes2keep]

    #         G_branch_ds = deepcopy(G_branch)
    #         G_branch_ds.remove_nodes_from(nodes2remove)
    #         G_branch_ds.remove_edges_from(list(G_branch.edges))
    #         for node1, node2 in zip(nodes2keep[:-1], nodes2keep[1:]):
    #             G_branch_ds.add_edge(node1, node2)
    #         G_branch_ds.fit_spline_tree_invariant(k=1)
    #         spline_tree_branch_ds = G_branch_ds.spline_tree

    #         # Compute sampling length from downsampled branch
    #         spline = spline_tree_branch_ds.nodes[0]["spline"]
    #         u = spline[1]
    #         tck = spline[0]
    #         pts = splev(u, tck)
    #         pts = np.stack(pts, axis=1)
    #         av_sample_distance = np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1))

    #         # Find dense line points
    #         tck_line, _ = splprep(pts.T, k=1, s=0, u=u)
    #         u_dense = np.arange(u[0], u[-1], sampling)
    #         u_dense = np.append(u_dense, u[-1])
    #         pts_line = splev(u_dense, tck_line)
    #         pts_line = np.stack(pts_line, axis=1)
    #         dense_line_pts = ct.evaluate(pts_line)


    #         # transform the branch
    #         G_branch_ds_transformed = deepcopy(G_branch_ds)
    #         G_branch_ds_transformed = transform_geometricgraph(
    #             G_branch_ds_transformed, ct, deriv_method="two-sided"
    #         )
    #         spline_tree_transformed_ds = G_branch_ds_transformed.spline_tree


    #         if len(spline_tree_transformed_ds.nodes) != 1:
    #             raise ValueError("transformed spline tree does not have 1 branch")
            

    #         # Find transformed knots
    #         spline = spline_tree_transformed_ds.nodes[0]["spline"]
    #         chspline = spline[0]
    #         u = spline[1]
    #         u_first_order = np.arange(u[0], u[-1], sampling)
    #         u_first_order = np.append(u_first_order, u[-1])
    #         trans_pts = chspline(u)


    #         # print("0th Order Mapping...")
    #         u_line = compute_parameterization(trans_pts)
    #         tck_line, u_line = splprep(trans_pts.T, k=1, s=0, u=u_line)
    #         u_line = np.arange(u_line[0], u_line[-1], sampling)
    #         u_line = np.append(u_line, u[-1])
    #         zero_order_pts = splev(u_line, tck_line)
    #         zero_order_pts = np.stack(zero_order_pts, axis=1)

    #         # print("1st order mapping...")
    #         first_order_pts = chspline(u_first_order)

    #         for method, method_pts in tqdm(
    #             zip(["Zeroth Order", "First Order"], [zero_order_pts, first_order_pts]),
    #             total=2,
    #             leave=False,
    #             disable=True,
    #         ):
    #             if len(dense_line_pts) < np.inf:
    #                 if dense_line_pts.shape[0] > 1000 or method_pts.shape[0] > 1000:
    #                     print(
    #                         f"{nid} comparing {dense_line_pts.shape[0]} with {method_pts.shape[0]}"
    #                     )
    #                 error = frechet_dist(dense_line_pts, method_pts)

    #                 res_ds_factors.append(ds_factor)
    #                 res_methods.append(method)
    #                 res_errors.append(error)
    #                 res_av_sample_distances.append(av_sample_distance)
    #                 res_neurons.append(swc)
    #                 res_spacings.append(spacing)
            


for sigma in sigmas:
    xv, phii = diffeo_gen_ara(sigma)
    ct = Diffeomorphism_Transform(xv, phii)

    transform_data = {"xv": xv, "phii": phii}
    transform_fname = swc_dir.parents[0] / f"exp-morpho-diffeo-{sigma}.pickle"
    with open(transform_fname, "wb") as handle:
        pickle.dump(transform_data, handle)

    Parallel(n_jobs=1)(
        delayed(process_swc)(sigma, ct, swc_path)
        for swc_path in tqdm(swc_paths, desc=f"mapping neurons at spacing {sigma}...")
    )

