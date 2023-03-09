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
import random


max_length = 5000
iterations = 100

fname = f"/cis/home/tathey/projects/mouselight/axon_mapping/random-lines-exp/random-lines_len-{max_length}_iter-{iterations}.pickle"

ct = CloudReg_Transform(
    "/cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed_ch1_otsu_iso_registration/downloop_1_v.mat",
    "/cis/home/tathey/projects/mouselight/axon_mapping/low_res/2018-12-01/precomputed_ch1_otsu_iso_registration/downloop_1_A.mat",
)


len_errors = []
bbox = [(np.amin(coords), np.amax(coords)) for coords in ct.og_coords]
print(bbox)
spacing = 1

for i in tqdm(range(iterations)):
    length = 5001
    while length > max_length:
        pt1 = [random.uniform(interval[0], interval[1]) for interval in bbox]
        pt2 = [random.uniform(interval[0], interval[1]) for interval in bbox]
        length = np.linalg.norm(np.subtract(pt1, pt2))
        print(length)

    dict = {"x": [pt1[0], pt2[0]], "y": [pt1[1], pt2[1]], "z": [pt1[2], pt2[2]], "sample": [0,1], "parent": [-1, 0]}
    df = pd.DataFrame(data=dict)
    G_branch = GeometricGraph(df=df, root=s[0])

    G_branch.fit_spline_tree_invariant()
    spline_tree_branch = G_branch.spline_tree

    # transform the branch
    G_branch_transformed = deepcopy(G_branch)
    G_branch_transformed = transform_geometricgraph(G_branch_transformed, ct, deriv_method="two-sided")

    spline_tree_transformed = G_branch_transformed.spline_tree

    # Compute sampling length from downsampled branch
    spline = spline_tree_branch.nodes[0]["spline"]
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
    spline = spline_tree_transformed.nodes[0]["spline"]
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

    z_error = frechet_dist(dense_line_pts, zero_order_pts)
    f_error = frechet_dist(dense_line_pts, first_order_pts)
    len_errors.append((length, z_error, f_error))


with open(fname, "wb") as handle:
    pickle.dump(len_errors, handle)