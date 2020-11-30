import numpy as np
import brainlit
from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
from brainlit.utils import swc
from cloudvolume.exceptions import SkeletonDecodeError
from scipy.interpolate import BSpline, splev
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import os
import pandas as pd

# EXTRACT DATA DIR FROM RELATIVE NOTEBOOK PATH
data_dir = Path(__file__).parents[2] / "data" / "poster_reproduction"
segments_dir = os.path.join(data_dir, "segments")
segments_swc_dir = os.path.join(data_dir, "segments_swc")
splines_dir = os.path.join(data_dir, "splines")

url_segments = "s3://open-neurodata/brainlit/brain1_segments"
mip = 0


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
mean_torsions = []
mean_curvatures = []
inconsistent_neurons = []
for i in np.arange(0, max_id):
    root_out_degree = 0
    i = int(i)
    with controlled_read_s3():
        segment_path = os.path.join(segments_dir, "{}.csv".format(i))
        if os.path.exists(segment_path) is False:
            df_neuron = swc.read_s3(url_segments, seg_id=i, mip=0)
            df_neuron.to_csv(segment_path)
        else:
            df_neuron = pd.read_csv(segment_path)

        string_id = str(i).zfill(3)
        swc_path = os.path.join(
            segments_swc_dir, "2018-08-01_G-{}_consensus.swc".format(string_id)
        )
        df_swc_offset_neuron, _, _, _ = swc.read_swc_offset(swc_path)

        print("Loaded seg_id={}, swc_path={}".format(i, swc_path))

        samples = df_neuron["sample"].to_numpy()
        parents = df_neuron["parent"].to_numpy()
        edges = np.array([samples, parents]).T
        sorted_edges = edges[np.argsort(edges[:, 0])]
        assert edges[0][0] == samples[0] and edges[0][1] == parents[0]

        swc_samples = df_swc_offset_neuron["sample"].to_numpy()
        swc_parents = df_swc_offset_neuron["parent"].to_numpy()
        swc_edges = np.array([swc_samples, swc_parents]).T
        sorted_swc_edges = swc_edges[np.argsort(swc_edges[:, 0])]
        assert swc_edges[0][0] == swc_samples[0] and swc_edges[0][1] == swc_parents[0]

        # print("Sorted edges from S3\n", sorted_edges[:10])
        # print("Sorted edges from .swc\n", sorted_swc_edges[:10])

        for parent in swc_parents:
            if len(np.where(swc_samples == parent)[0]) == 0:
                if parent == -1:
                    root_out_degree += 1
            if len(np.where(swc_parents == parent)[0]) > 1:
                print("Duplicate parent: {}".format(parent))

        if root_out_degree > 1:
            inconsistent_neurons.append(i)

print("Found inconsistent neurons: {}".format(inconsistent_neurons))
