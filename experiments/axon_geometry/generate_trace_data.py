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


class controlled_read_s3:
    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        return isinstance(value, SkeletonDecodeError)


def generate_brain_trace_data(brain: str):
    root_dir = data_dir = Path(__file__).parents[2]
    data_dir = os.path.join(root_dir, "data/axon_geometry/{}".format(brain))
    experiment_dir = os.path.join(root_dir, "experiments/axon_geometry")
    segments_dir = os.path.join(data_dir, "segments")
    segments_swc_dir = os.path.join(data_dir, "segments_swc")
    # splines_dir = os.path.join(data_dir, "splines")
    # curvatures_dir = os.path.join(data_dir, "curvatures")
    # orsions_dir = os.path.join(data_dir, "torsions")
    trace_data_dir = os.path.join(data_dir, "trace_data")
    print(segments_dir)

    url_segments = "s3://open-neurodata/brainlit/{}_segments".format(brain)

    max_id = 300
    # mip = 0
    # ids = []
    # seg_lengths = []
    # mean_torsions = []
    # mean_curvatures = []
    # trace_data = []
    for i in np.arange(0, max_id):
        i = int(i)

        string_id = str(i).zfill(3)
        swc_path = os.path.join(
            segments_swc_dir,
            "{}_G-{}_consensus.swc".format(
                "2018-08-01" if brain == "brain1" else "2018-12-01", string_id
            ),
        )
        if os.path.exists(swc_path):
            df_swc_offset_neuron, _, _, _ = swc.read_swc_offset(swc_path)

            # curvature_path = os.path.join(curvatures_dir, "{}.npy".format(i))
            # torsion_path = os.path.join(torsions_dir, "{}.npy".format(i))
            trace_data_path = os.path.join(trace_data_dir, "{}.npy".format(i))

            print("Loaded segment {}".format(i))

            # if os.path.exists(curvature_path) is True:
            #     curvature_data = np.load(curvature_path)
            # if os.path.exists(torsion_path) is True:
            #     torsion_data = np.load(torsion_path)

            # if (
            #     os.path.exists(curvature_path) is False
            #     or os.path.exists(torsion_path) is False
            # ):
            G = GeometricGraph(df=df_swc_offset_neuron)
            print("Initialized GeometricGraph")
            spline_tree = G.fit_spline_tree_invariant()
            print("Computed splines")

            # curvature_data = np.zeros((len(spline_tree.nodes), 2))
            # torsion_data = np.zeros((len(spline_tree.nodes), 2))
            trace_data = np.empty(len(spline_tree.nodes), dtype="object")
            for j, node in enumerate(spline_tree.nodes):
                spline = spline_tree.nodes[node]
                starting_length = spline["starting_length"]
                path = spline["path"]
                tck, u_nm = spline["spline"]

                t = tck[0]
                c = tck[1]
                k = tck[2]

                # convert nm in um
                u_um = u_nm
                # evaluate segment length (in um)
                seg_length = u_um[-1] - u_um[0]
                # curvature_data[j, 0] = seg_length
                # torsion_data[j, 0] = seg_length
                # resample points at 1um
                uu = np.arange(u_um[0], u_um[-1] + 0.9, 1)
                # evaluate mean curvature of the segment
                _curvature = curvature(uu, t, c, k)
                mean_curvature = np.mean(_curvature)
                # curvature_data[j, 1] = mean_curvature
                # evaluate mean torsion of the segment
                _torsion = torsion(uu, t, c, k)
                mean_torsion = np.mean(_torsion)
                # torsion_data[j, 1] = mean_torsion

                trace_data[j] = {
                    "seg_length": seg_length,
                    "starting_length": starting_length,
                    "mean_curvature": mean_curvature,
                    "mean_torsion": mean_torsion,
                    "curvature": _curvature,
                    "torsion": _torsion,
                }

            # np.save(curvature_path, curvature_data)
            # np.save(torsion_path, torsion_data)
            np.save(trace_data_path, trace_data)

for brain in ["brain1", "brain2"]:
    generate_brain_trace_data(brain)