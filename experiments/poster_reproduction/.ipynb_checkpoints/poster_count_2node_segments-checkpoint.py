import numpy as np
import brainlit
import scipy
import scipy.stats
from brainlit.utils import swc
from cloudvolume.exceptions import SkeletonDecodeError
from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
from brainlit.algorithms.trace_analysis.spline_fxns import curvature, torsion
import os
from pathlib import Path
import pandas as pd
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

# EXTRACT DATA DIR FROM RELATIVE NOTEBOOK PATH
brain = "brain2"
root_dir = data_dir = Path(__file__).parents[2]
data_dir = os.path.join(root_dir, "data/poster_reproduction/{}".format(brain))
experiment_dir = os.path.join(root_dir, "experiments/poster_reproduction")
segments_dir = os.path.join(data_dir, "segments")
segments_swc_dir = os.path.join(data_dir, "segments_swc")
splines_dir = os.path.join(data_dir, "splines")
curvatures_dir = os.path.join(data_dir, "curvatures")
torsions_dir = os.path.join(data_dir, "torsions")
print(segments_dir)

url_segments = "s3://open-neurodata/brainlit/{}_segments".format(brain)


class controlled_read_s3:
    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return isinstance(value, SkeletonDecodeError)


# DOWNLOAD (OR READ) SEGMENTS THEN COMPUTE AND SAVE SPLINES
max_id = 300
mip = 0
ids = []
seg_lengths = []
zero_curvature_path_lengths = []
zero_torsion_path_lengths = []

for i, file in enumerate(os.listdir(segments_swc_dir)):
    print(i, file)
    swc_path = os.path.join(segments_swc_dir, file)
    df_swc_offset_neuron, _, _, _ = swc.read_swc_offset(swc_path)
    print("Loaded segment {}".format(i))
    G = GeometricGraph(df=df_swc_offset_neuron)
    print("Initialized GeometricGraph")
    spline_tree = G.fit_spline_tree_invariant()
    print("Computed spline tree")
    for j, node in enumerate(spline_tree.nodes):
        spline = spline_tree.nodes[node]
        path = spline["path"]
        tck, u_nm = spline["spline"]

        t = tck[0]
        c = tck[1]
        k = tck[2]

        # convert nm in um
        u_um = u_nm
        # evaluate segment length (in um)
        # resample points at 1um
        uu = np.arange(u_um[0], u_um[-1] + 0.9, 1)
        # evaluate mean curvature of the segment
        _curvature = curvature(uu, t, c, k)
        mean_curvature = np.mean(_curvature)
        if mean_curvature < 1e-11:
            zero_curvature_path_lengths.append(len(path))
        _torsion = torsion(uu, t, c, k)
        mean_torsion = np.abs(np.mean(_torsion))
        if mean_torsion < 1e-11:
            zero_torsion_path_lengths.append(len(path))

zero_curvature_path_lengths = np.array(zero_curvature_path_lengths)
zero_torsion_path_lengths = np.array(zero_torsion_path_lengths)
r_curvature = len(np.where(zero_curvature_path_lengths == 2)[0]) / len(zero_curvature_path_lengths)
r_torsion = len(np.where(zero_torsion_path_lengths <= 3)[0]) / len(zero_torsion_path_lengths)
print(r_curvature, r_torsion)