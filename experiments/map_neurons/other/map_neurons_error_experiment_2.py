from brainlit.map_neurons.diffeo_gen import expR
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

# INPUTS
first_neuron = 0
spacings = [3.0, 6.0, 12.0, 24.0]
sampling = 2
ds_factors = [1, 2, 4, 8, 16]
swc_dir = Path("/cis/home/tathey/projects/mouselight/axon_mapping/ds_experiment/mouselight-swcs")

swcs = os.listdir(swc_dir)

print(
    f"Processing {len(swcs)} neurons with a sampling rate of {sampling} and downsampling factor {ds_factors} with spacings of {spacings} "
)

for spacing in tqdm(spacings, desc="Spacings..."):
    """
    CREATE DIFFEOMORPHISM
    """
    # a domain for sampling your velocity and deformatoin
    dv = np.array([spacing]*3)
    nv = np.array([132,80,114])
    xv = [np.arange(n)*d - (n-1)*d/2 for n,d in zip(nv,dv)]
    levels = np.linspace(-40*int(spacing),40*int(spacing),10)


    XV = torch.stack(torch.meshgrid([torch.as_tensor(x) for x in xv],indexing='ij'),-1)

    # a frequency domain
    fv = [np.arange(n)/n/d for n,d in zip(nv,dv)]
    FV = np.stack(np.meshgrid(*fv,indexing='ij'),-1)
    a = spacing
    p = 2.0
    LL = (1.0 - 2.0*a**2*np.sum(((np.cos(2.0*np.pi*FV*dv)   - 1))/dv**2,-1))**(2*p)
    K = 1.0/LL
    fig,ax = plt.subplots()
    ax.imshow(np.fft.ifftshift(K[0]))
    ax.set_title('smoothing kernel')


    # lets make a new p which is really simple for testing
    # sample white noise
    Lm = np.random.randn(*FV.shape)*200

    # smooth it with sqrt(K) (here I smoothed with K to be a bit smoother)
    v = np.fft.ifftn(np.fft.fftn(Lm,axes=(0,1,2))*K[...,None],axes=(0,1,2)).real

    #shoot it with remannian exponential
    phii = expR([torch.tensor(x) for x in xv],torch.tensor(v),K,n=10)
    phii = phii.detach().cpu().numpy()

    ct = Diffeomorphism_Transform(xv, phii)

    transform_data = {"xv": xv, "phii": phii}
    transform_fname = swc_dir.parents[0] / f"{spacing}.pickle"
    with open(transform_fname, "wb") as handle:
        pickle.dump(transform_data, handle)


    """
    MAP NEURONS
    """
    res_ds_factors = []
    res_methods = []
    res_errors = []
    res_av_sample_distances = []
    res_neurons = []
    res_spacings = []
    for ns, swc in enumerate(tqdm(swcs, desc="different neurons", leave=False)):
        if ns < first_neuron or "swc" not in swc:
            continue

        swc_path = swc_dir / swc
        ntrace = NeuronTrace(str(swc_path), rounding=False)
        g = ntrace.get_graph()

        coords = []
        for n in g.nodes:
            coords.append([g.nodes[n][i] for i in ['x','y','z']])

        
        dupes = []
        seen = set()
        for coord in coords:
            coord = tuple(coord)
            if coord in seen:
                dupes.append(coord)
            else:
                seen.add(coord)
        if len(dupes) > 0:
            print(f"Duplicate nodes in {swc}")
            continue

        coords = np.array(coords)
        mx = np.amax(coords, axis=0)
        mn = np.amin(coords, axis=0)
        center = np.mean(np.array([mx,mn]), axis=0)
        # center = np.mean(coords,axis=0)

        G_neuron = GeometricGraph()
        for n in g.nodes:
            G_neuron.add_node(n, loc = np.array([g.nodes[n][i] for i in ['x','y','z']]) - center)
        for e in g.edges:
            G_neuron.add_edge(e[0], e[1])
        spline_tree = G_neuron.fit_spline_tree_invariant()
        for ds_factor in tqdm(ds_factors, desc="downsample factors...", leave=False, disable=True):
            for branch_id in tqdm(
                spline_tree.nodes, desc="Processing branches...", leave=False, disable=True
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
                spline = spline_tree_branch_ds.nodes[0]["spline"]
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

                for method, method_pts in tqdm(zip(
                    ["Zeroth Order", "First Order"], [zero_order_pts, first_order_pts]
                ), total=2, leave=False, disable=True):
                    if len(dense_line_pts) < np.inf:
                        error = frechet_dist(dense_line_pts, method_pts)

                        res_ds_factors.append(ds_factor)
                        res_methods.append(method)
                        res_errors.append(error)
                        res_av_sample_distances.append(av_sample_distance)
                        res_neurons.append(swc)
                        res_spacings.append(spacing)


        fname = f"/cis/home/tathey/projects/mouselight/axon_mapping/ds_experiment/mouselight-swcs/spac{spacing}_{ns}neurons_sample{sampling}.pickle"
        data = {
            "Method": res_methods,
            "Frechet Error (microns)": res_errors,
            "Average Sampling (microns)": res_av_sample_distances,
            "Downsample factor": res_ds_factors,
            "Sample": res_neurons,
            "Spacing": res_spacings
        }
        with open(fname, "wb") as handle:
            pickle.dump(data, handle)