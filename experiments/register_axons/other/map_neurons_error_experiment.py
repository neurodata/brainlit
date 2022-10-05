from pathlib import Path
from brainlit.utils.Neuron_trace import NeuronTrace
from brainlit.algorithms.trace_analysis.fit_spline import (
    GeometricGraph,
    compute_parameterization,
)
from scipy.interpolate import splprep, BSpline, CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.interpolate import splev
from brainlit.map_neurons.map_neurons import (
    DiffeomorphismAction,
    transform_geometricgraph,
    compute_derivs,
)
import pandas as pd
import numpy as np
import h5py
from brainlit.map_neurons.map_neurons import CloudReg_Transform
from scipy.spatial.distance import cosine
from tqdm import tqdm
from cloudvolume import CloudVolume
from similaritymeasures import frechet_dist
import seaborn as sns
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation
import time
from cloudvolume.exceptions import SkeletonDecodeError
import pickle
from copy import deepcopy


vol = CloudVolume(
    "precomputed://file:///cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed/axons"
)
shp = np.array(vol.shape)
res_im = np.array(vol.resolution) / 1000
origin_im = (shp[:3] - 1) * res_im / 2

valid_ids = []
for i in range(100):
    try:
        vol.skeleton.get(i)
    except SkeletonDecodeError:
        continue
    valid_ids.append(i)

print(f"Valid IDs: {valid_ids}")

methods = []
errors = []
av_sample_distances = []

spacing = 2
inter = -1
ds_factor = 1

print(
    f"Processing neurons starting with {inter} with a spacing {spacing} and downsampling factor {ds_factor}"
)

ct = CloudReg_Transform(
    "/cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed_ch1_otsu_iso_registration/downloop_1_v.mat",
    "/cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed_ch1_otsu_iso_registration/downloop_1_A.mat",
)

for id in tqdm(valid_ids, desc="Processing neurons..."):
    if id <= inter:
        continue

    # get coords in proper cooordinates
    skel = vol.skeleton.get(id)
    coords = skel.vertices / 1000 - origin_im

    # apply affine transform
    coords = ct.apply_affine(coords)

    G_neuron = GeometricGraph()
    for node_id, coord in enumerate(coords):
        G_neuron.add_node(node_id, loc=coord)
    for edge in skel.edges:
        G_neuron.add_edge(edge[0], edge[1])

    if id == 20:
        G_neuron.remove_node(4957)
        G_neuron.add_edge(4956, 4958)
    if id == 53:
        G_neuron.remove_node(0)

    spline_tree = G_neuron.fit_spline_tree_invariant()

    # For each branch
    for branch_id in tqdm(
        spline_tree.nodes, desc="Processing branches...", leave=False
    ):
        # Create geometric graph for the branch
        path = spline_tree.nodes[branch_id]["path"]
        x = []
        y = []
        z = []
        s = []
        p = [-1]

        for point_num, point_id in enumerate(path):
            loc = G_neuron.nodes[point_id]["loc"]
            x.append(loc[0])
            y.append(loc[1])
            z.append(loc[2])
            s.append(point_num)
            if point_num > 0:
                p.append(point_num - 1)

        dict = {"x": x, "y": y, "z": z, "sample": s, "parent": p}
        df = pd.DataFrame(data=dict)
        G_branch = GeometricGraph(df=df, root=s[0])
        G_branch.fit_spline_tree_invariant()
        spline_tree_branch = G_branch.spline_tree

        # downsample
        nodes2keep = [node_i for node_i in range(0, len(s), ds_factor)]
        if s[-1] not in nodes2keep:
            nodes2keep += [s[-1]]
        nodes2remove = [node_i for node_i in s if node_i not in nodes2keep]

        path = spline_tree_branch.nodes[0]["path"]
        tck, us = spline_tree_branch.nodes[0]["spline"]
        positions = np.array(splev(us, tck, der=0)).T

        G_branch_ds = deepcopy(G_branch)
        G_branch_ds.remove_nodes_from(nodes2remove)
        G_branch_ds.remove_edges_from(list(G_branch.edges))
        for node1, node2 in zip(nodes2keep[:-1], nodes2keep[1:]):
            G_branch_ds.add_edge(node1, node2)
        G_branch_ds.fit_spline_tree_invariant(k=1)
        spline_tree_branch_ds = G_branch_ds.spline_tree

        # transform the branch
        G_branch_transformed = deepcopy(G_branch)
        G_branch_ds_transformed = deepcopy(G_branch_ds)
        # G_branch_transformed = transform_geometricgraph(G_branch_transformed, ct, deriv_method="difference")
        G_branch_ds_transformed = transform_geometricgraph(
            G_branch_ds_transformed, ct, deriv_method="two-sided"
        )

        spline_tree_transformed = G_branch_transformed.spline_tree
        spline_tree_transformed_ds = G_branch_ds_transformed.spline_tree

        if len(spline_tree_transformed_ds.nodes) != 1:
            raise ValueError("transformed spline tree does not have 1 branch")

        # Compute sampling length from downsampled branch
        spline = spline_tree_branch_ds.nodes[0]["spline"]
        u = spline[1]
        tck = spline[0]
        pts = splev(u, tck)
        pts = np.stack(pts, axis=1)
        av_sample_distance = np.mean(np.linalg.norm(np.diff(pts, axis=0), axis=1))

        # Access original knots and compute sample distance
        spline = spline_tree_branch.nodes[0]["spline"]
        u = spline[1]
        tck = spline[0]
        pts = splev(u, tck)
        pts = np.stack(pts, axis=1)

        # Find dense line points
        tck_line, _ = splprep(pts.T, k=1, s=0, u=u)
        u_dense = np.arange(u[0], u[-1], spacing)
        u_dense = np.append(u_dense, u[-1])
        pts_line = splev(u_dense, tck_line)
        pts_line = np.stack(pts_line, axis=1)
        dense_line_pts = ct.evaluate(pts_line)

        # Find transformed knots
        spline = spline_tree_transformed_ds.nodes[0]["spline"]
        chspline = spline[0]
        u = spline[1]
        u_first_order = np.arange(u[0], u[-1], spacing)
        u_first_order = np.append(u_first_order, u[-1])
        trans_pts = chspline(u)

        # print("0th Order Mapping...")
        u_line = compute_parameterization(trans_pts)
        tck_line, u_line = splprep(trans_pts.T, k=1, s=0, u=u_line)
        u_line = np.arange(u_line[0], u_line[-1], spacing)
        u_line = np.append(u_line, u[-1])
        zero_order_pts = splev(u_line, tck_line)
        zero_order_pts = np.stack(zero_order_pts, axis=1)

        # print("1st order mapping...")
        first_order_pts = chspline(u_first_order)

        for method, method_pts in zip(
            ["Zeroth Order", "First Order"], [zero_order_pts, first_order_pts]
        ):
            error = frechet_dist(dense_line_pts, method_pts)

            av_sample_distances.append(av_sample_distance)
            errors.append(error)
            methods.append(method)

        if errors[-2] - errors[-1] > 7:
            print(f"neuron {id} branch {branch_id} error {errors[-2] - errors[-1]}")

    fname = f"/cis/home/tathey/projects/mouselight/axon_mapping/ds_experiment/derivdiff2_errsthru{id}_spac{spacing}_ds{ds_factor}.pickle"
    data = {
        "Method": methods,
        "Frechet Distance": errors,
        "Average Sampling": av_sample_distances,
    }
    with open(fname, "wb") as handle:
        pickle.dump(data, handle)
