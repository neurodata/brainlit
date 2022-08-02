from pathlib import Path
from brainlit.utils.Neuron_trace import NeuronTrace
from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph, compute_parameterization
from scipy.interpolate import splprep, BSpline, CubicHermiteSpline
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy.interpolate import splev
from brainlit.map_neurons.map_neurons import DiffeomorphismAction, transform_GeometricGraph
import pandas as pd
import numpy as np
import h5py
from brainlit.map_neurons.map_neurons import CloudReg_Transform
from scipy.spatial.distance import cosine
from tqdm import tqdm
from cloudvolume import CloudVolume
from similaritymeasures import frechet_dist
from frechetdist import frdist
import seaborn as sns
from statannotations.Annotator import Annotator
from statannot import add_stat_annotation
import time
from cloudvolume.exceptions import SkeletonDecodeError
import pickle


vol = CloudVolume("precomputed://file:///cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed/axons")
shp = np.array(vol.shape)
res_im = np.array(vol.resolution)/1000
origin_im = (shp[:3] - 1) * res_im / 2

valid_ids = []
for i in range(100):
    try:
        vol.skeleton.get(i)
    except SkeletonDecodeError:
        continue
    valid_ids.append(i)

print(f"Valid IDs: {valid_ids}")

truths = []
errors = []
methods = []

spacing = 2
plot_branches = False
deriv_method = "difference"
method_name = "diff2"
inter = 0

ct = CloudReg_Transform("/cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed_ch1_otsu_iso_registration/downloop_1_v.mat",
"/cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed_ch1_otsu_iso_registration/downloop_1_A.mat")

big_error_ids = []

len1s = []
len2s = []

for id in tqdm(valid_ids, desc="Processing neurons..."):
    if id <= inter:
        continue

    # get coords in proper cooordinates
    skel = vol.skeleton.get(id)
    coords = skel.vertices/1000 - origin_im

    # apply affine transform
    coords = ct.apply_affine(coords)

    G = GeometricGraph()
    for node_id, coord in enumerate(coords):
        G.add_node(node_id, loc=coord)
    for edge in skel.edges:
        G.add_edge(edge[0], edge[1])

    if id == 20:
        G.remove_node(4957)
        G.add_edge(4956, 4958)
    if id == 53:
        G.remove_node(0)

    spline_tree = G.fit_spline_tree_invariant()
    G_tranformed = transform_GeometricGraph(G, ct, deriv_method=deriv_method)
    spline_tree_transformed = G_tranformed.spline_tree


    for i, node in enumerate(tqdm(spline_tree.nodes, desc="Processing branches...", leave=False)):
        spline = spline_tree.nodes[node]["spline"]
        u = spline[1]
        tck = spline[0]
        pts = splev(u, tck)
        pts = np.stack(pts, axis=1)

        #print("dense line points...")
        tck_line, _ = splprep(pts.T, k=1, s=0, u=u)
        u_dense = np.arange(u[0], u[-1], spacing)
        u_dense = np.append(u_dense, u[-1])
        pts_line = splev(u_dense, tck_line)
        pts_line = np.stack(pts_line, axis=1)
        dense_line_pts = ct.evaluate(pts_line)

        #print("dense spline points...")
        pts_spline = splev(u_dense, tck)
        pts_spline = np.stack(pts_spline, axis=1)
        dense_spline_pts = ct.evaluate(pts_spline)

        # Mappings
        spline = spline_tree_transformed.nodes[node]["spline"]
        chspline = spline[0]
        u = spline[1]
        u_first_order = np.arange(u[0], u[-1], spacing)
        u_first_order = np.append(u_first_order, u[-1])
        trans_pts = chspline(u)

        #print("0th Order Mapping...")
        u_line = compute_parameterization(trans_pts)
        tck_line, u_line = splprep(trans_pts.T, k=1, s=0, u=u_line)
        u_line = np.arange(u_line[0], u_line[-1]+spacing, spacing)
        u_line = np.append(u_line, u[-1])
        zero_order_pts = splev(u_line, tck_line)
        zero_order_pts = np.stack(zero_order_pts, axis=1)

        #print("1st order mapping...")
        first_order_pts = chspline(u_first_order)
        
        truths_list = {"Linear Spline Interpolation": dense_line_pts, "Cubic Spline Interpolation": dense_spline_pts}
        methods_list = {"0th Order Action": zero_order_pts, "1st Order Action": first_order_pts}

        for truthkey in tqdm(truths_list.keys(), desc="Different ground truths...", leave=False, disable=True):
            truth = truths_list[truthkey]
            for methodkey in tqdm(methods_list.keys(), desc="Different methods...", leave=False, disable=True):
                method = methods_list[methodkey]

                truths.append(truthkey)
                methods.append(methodkey)
                error = frechet_dist(truth, method) #frdist(truth, method)
                errors.append(error)

                if error > 10 and truthkey == "Linear Spline Interpolation":
                    big_error_ids.append((id, node))
        
        if plot_branches:
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(1, 1, 1, projection="3d")
            ax.plot(dense_line_pts[:,0], dense_line_pts[:,1], dense_line_pts[:,2], linestyle='-',linewidth=0.8, color='blue', label="Dense Points Lines")
            ax.plot(dense_spline_pts[:,0], dense_spline_pts[:,1], dense_spline_pts[:,2], linestyle='-',linewidth=0.8, color="orange", label="Dense Points Spline")
            ax.plot(trans_pts[:,0], trans_pts[:,1], trans_pts[:,2], linestyle='-',linewidth=0.8, label="0th Order Mapping", color='green')
            ax.plot(first_order_pts[:,0], first_order_pts[:,1], first_order_pts[:,2], linestyle='-',linewidth=0.8, label="1st Order Mapping", color='red')

            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.grid(True)
            ax.set_title("Template Space")
            ax.legend()

            plt.show()

    fname = f"/cis/home/tathey1/projects/mouselight/axon_mapping/figures/deriv{method_name}_errsthru{id}_spac{spacing}.pickle"
    data = {"Method": methods, "Frechet Distance": errors, "Ground Truth": truths}
    with open(fname, 'wb') as handle:
        pickle.dump(data, handle)