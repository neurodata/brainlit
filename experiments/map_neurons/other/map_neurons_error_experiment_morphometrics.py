from brainlit.map_neurons.diffeo_gen import diffeo_gen_ara
import numpy as np
from scipy.spatial.distance import cosine
from brainlit.map_neurons.map_neurons import (
    Diffeomorphism_Transform,
)
from brainlit.map_neurons.utils import replace_root, ZerothFirstOrderNeuron
import os
from pathlib import Path
from tqdm import tqdm
import os
import pickle
from tqdm import tqdm
from joblib import Parallel, delayed
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


def process_swc(sigma, ct, swc_path):
    dir = swc_path.parent
    stem = swc_path.stem
    fname = str(dir / "results" / stem)
    fname_gt = str(fname) + f"-sig-{sigma}-gt.swc"
    fname_0 = str(fname) + f"-sig-{sigma}-0.swc"
    fname_1 = str(fname) + f"-sig-{sigma}-1.swc"

    neuron = ngauge.Neuron.from_swc(swc_path)
    neuron = replace_root(neuron)
    neuron = check_duplicates_center(neuron)

    n = ZerothFirstOrderNeuron(neuron, ct, sampling=2)

    if not os.path.exists(fname_gt):
        neuron_gt = n.get_gt()
        neuron_gt.to_swc(fname_gt)

    if not os.path.exists(fname_0) or not os.path.exists(fname_1):
        neuron_0, neuron_1 = n.get_transforms()
        neuron_0.to_swc(fname_0)
        neuron_1.to_swc(fname_1)


for sigma in sigmas:
    transform_fname = swc_dir.parents[0] / f"exp-morpho-diffeo-{sigma}.pickle"
    if os.path.exists(transform_fname):
        with open(transform_fname, "rb") as handle:
            transform_data = pickle.load(handle)
            xv = transform_data["xv"]
            phii = transform_data["phii"]
    else:
        xv, phii = diffeo_gen_ara(sigma)
        transform_data = {"xv": xv, "phii": phii}
        transform_fname = swc_dir.parents[0] / f"exp-morpho-diffeo-{sigma}.pickle"
        with open(transform_fname, "wb") as handle:
            pickle.dump(transform_data, handle)

    ct = Diffeomorphism_Transform(xv, phii)

    Parallel(n_jobs=4)(
        delayed(process_swc)(sigma, ct, swc_path)
        for swc_path in tqdm(swc_paths, desc=f"mapping neurons at spacing {sigma}...")
    )
