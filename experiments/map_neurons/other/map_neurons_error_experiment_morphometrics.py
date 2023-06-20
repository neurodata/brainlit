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
from brainlit.map_neurons.utils import resample_neuron, replace_root, zeroth_order_map_neuron, ZerothFirstOrderNeuron
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

def check_duplicates_center(neuron):
    assert len(neuron.branches) == 1

    stack = []
    stack += neuron.branches[0]
    coords = []

    while len(stack) > 0:
        child = stack.pop()
        stack += child.children
        coords.append([child.x, child.y, child.z])

    
    # look for duplicates
    dupes = []
    seen = set()
    for coord in coords:
        coord = tuple(coord)
        if coord in seen:
            dupes.append(coord)
        else:
            seen.add(coord)

    # center coordinates
    if len(dupes) > 0:
        raise ValueError(f"Duplicate nodes")
    else:
        coords = np.array(coords)
        mx = np.amax(coords, axis=0)
        mn = np.amin(coords, axis=0)
        center = np.mean(np.array([mx, mn]), axis=0)
        stack = [neuron.branches[0]]

        while len(stack) > 0:
            child = stack.pop()
            stack += child.children

            child.x -= center[0]
            child.y -= center[1]
            child.z -= center[2]
        
    return neuron






def process_swc(max_r, ct, swc_path):
    neuron = ngauge.Neuron.from_swc(swc_path)
    neuron = replace_root(neuron)
    neuron = check_duplicates_center(neuron)

    # neuron_us = resample_neuron(neuron, sampling=2)
    # neuron_us = zeroth_order_map_neuron(neuron_us, ct)
    # neuron_us.to_swc(dir / "results"  / f"{stem}-gt.swc")

    # # Zeroth order mapping
    # neuron = ngauge.Neuron.from_swc(swc_path)
    # neuron = replace_root(neuron)
    # neuron = check_duplicates_center(neuron)

    # neuron = zeroth_order_map_neuron(neuron, ct)
    # neuron.to_swc(dir / "results"  / f"{stem}-0.swc")

    n = ZerothFirstOrderNeuron(neuron, ct, sampling=2)

    dir = swc_path.parent
    stem = swc_path.stem
    fname = str(dir / "results" / stem)
    neuron_gt = n.get_gt()
    neuron_gt.to_swc(str(fname) + "-gt.swc")

    neuron_0, neuron_1 = n.get_transforms()
    neuron_0.to_swc(str(fname) + "-0.swc")
    neuron_1.to_swc(str(fname) + "-1.swc")

    raise ValueError()
            


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

