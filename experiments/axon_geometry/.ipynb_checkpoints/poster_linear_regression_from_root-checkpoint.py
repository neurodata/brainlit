import numpy as np
import brainlit
import scipy
import scipy.stats
from brainlit.utils import swc
from cloudvolume.exceptions import SkeletonDecodeError
from brainlit.algorithms.trace_analysis.fit_spline import GeometricGraph
from brainlit.algorithms.trace_analysis.spline_fxns import curvature, torsion
import os
import sys
from pathlib import Path
import pandas as pd
from networkx.readwrite import json_graph
import json
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "9"

device = torch.device("cuda:0" if torch.cuda.is_available() == True else "cpu")
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# EXTRACT DATA DIR FROM RELATIVE NOTEBOOK PATH
brain = "brain2"
root_dir = data_dir = Path(os.path.join(os.getcwd(), __file__)).parents[2]
data_dir = os.path.join(root_dir, "data/poster_reproduction/{}".format(brain))
experiment_dir = os.path.join(root_dir, "experiments/poster_reproduction")
segments_dir = os.path.join(data_dir, "segments")
segments_swc_dir = os.path.join(data_dir, "segments_swc")
# splines_dir = os.path.join(data_dir, "splines")
# curvatures_dir = os.path.join(data_dir, "curvatures")
# orsions_dir = os.path.join(data_dir, "torsions")
trace_data_dir = os.path.join(data_dir, "trace_data")
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
mean_torsions = []
mean_curvatures = []
d_from_root = []
torsions = []
curvatures = []
for i in np.arange(0, max_id):
    i = int(i)
    with controlled_read_s3():
        segment_path = os.path.join(segments_dir, "{}.csv".format(i))
        if os.path.exists(segment_path) is False:
            df_neuron = swc.read_s3(url_segments, seg_id=i, mip=0)
            df_neuron.to_csv(segment_path)
        else:
            df_neuron = pd.read_csv(segment_path)

        string_id = str(i).zfill(3)
        swc_path = [os.path.join(
            segments_swc_dir,
            "{}_G-{}_{}onsensus.swc".format(
                "2018-08-01" if brain == "brain1" else "2018-12-01", string_id, c
            ),
        ) for c in ["c", "C"]]
        
        for try_path in swc_path:
            if os.path.exists(try_path):
                df_swc_offset_neuron, _, _, _ = swc.read_swc_offset(try_path)

        # curvature_path = os.path.join(curvatures_dir, "{}.npy".format(i))
        # torsion_path = os.path.join(torsions_dir, "{}.npy".format(i))
        trace_data_path = os.path.join(trace_data_dir, "{}.npy".format(i))

        print("Loaded segment {}".format(i))

        # if os.path.exists(curvature_path) is True:
        #     curvature_data = np.load(curvature_path)
        # if os.path.exists(torsion_path) is True:
        #     torsion_data = np.load(torsion_path)
        if os.path.exists(trace_data_path) is True:
            trace_data = np.load(trace_data_path, allow_pickle=True)

            for node in trace_data:
                # print(node)
                seg_lengths.append(node["seg_length"])
                mean_curvatures.append(node["mean_curvature"])
                mean_torsions.append(node["mean_torsion"])
                _curvatures = node["curvature"]
                _torsions = node["torsion"]
                assert len(_curvatures) == len(_torsions)
                m = len(_curvatures)
                # print(node["starting_length"])
                d_from_root.append(node["starting_length"] + np.arange(0, m))
                # d_from_root.append(np.arange(0, m)/node["seg_length"])
                torsions.append(_torsions)
                curvatures.append(_curvatures)

        # if (
        #     os.path.exists(trace_data_path) is False
        #     os.path.exists(curvature_path) is False
        #     or os.path.exists(torsion_path) is False
        # ):
        #     G = GeometricGraph(df=df_swc_offset_neuron)
        #     print("Initialized GeometricGraph")
        #     spline_tree = G.fit_spline_tree_invariant()
        #     print("Computed splines")

        #     curvature_data = np.zeros((len(spline_tree.nodes), 2))
        #     torsion_data = np.zeros((len(spline_tree.nodes), 2))
        #     for j, node in enumerate(spline_tree.nodes):
        #         spline = spline_tree.nodes[node]
        #         starting_length = spline["starting_length"]
        #         path = spline["path"]
        #         tck, u_nm = spline["spline"]

        #         t = tck[0]
        #         c = tck[1]
        #         k = tck[2]

        #         # convert nm in um
        #         u_um = u_nm
        #         # evaluate segment length (in um)
        #         seg_length = u_um[-1] - u_um[0]
        #         curvature_data[j, 0] = seg_length
        #         torsion_data[j, 0] = seg_length
        #         # resample points at 1um
        #         uu = np.arange(u_um[0], u_um[-1] + 0.9, 1)
        #         # evaluate mean curvature of the segment
        #         _curvature = curvature(uu, t, c, k)
        #         mean_curvature = np.mean(_curvature)
        #         # curvature_data[j, 1] = mean_curvature
        #         # evaluate mean torsion of the segment
        #         _torsion = torsion(uu, t, c, k)
        #         mean_torsion = np.mean(_torsion)
        #         # torsion_data[j, 1] = mean_torsion

        #         trace_data[j] = {
        #             "seg_length": seg_length,
        #             "starting_length": starting_length,
        #             "mean_curvature": mean_curvature,
        #             "mean_torsion": mean_torsion,
        #             "curvature": _curvature,
        #             "torsion": _torsion
        #         }

        #     # np.save(curvature_path, curvature_data)
        #     # np.save(torsion_path, torsion_data)
        #     np.save(trace_data_path, trace_data)

        # seg_lengths.append(curvature_data[:, 0])
        # mean_curvatures.append(curvature_data[:, 1])
        # mean_torsions.append(torsion_data[:, 1])

# seg_lengths = np.concatenate(seg_lengths)
seg_lengths = np.array(seg_lengths)
mean_curvatures = np.array(mean_curvatures)
mean_torsions = np.array(mean_torsions)
d_from_root = np.concatenate(d_from_root)
curvatures = np.concatenate(curvatures)
torsions = np.concatenate(torsions)
# remove all root points
masked_d_from_root = np.ma.masked_less(d_from_root, 1e-16)
d_from_root = masked_d_from_root.compressed()
assert len(np.where(d_from_root < 1e-16)[0]) == 0
curvatures = curvatures[~masked_d_from_root.mask]
torsions = torsions[~masked_d_from_root.mask]

nonzero_curvatures_masked = np.ma.masked_less(
    curvatures, 1e-16
)
nonzero_curvatures = nonzero_curvatures_masked.compressed()
nonzero_curvatures_d_from_root = d_from_root[
    nonzero_curvatures_masked.mask == 0
]
nonzero_torsions_masked = np.ma.masked_less(
    torsions, 1e-16
)
nonzero_torsions = nonzero_torsions_masked.compressed()
nonzero_torsions_d_from_root = d_from_root[nonzero_torsions_masked.mask == 0]

log_curvatures = np.log10(nonzero_curvatures)
log_torsions = np.log10(np.abs(nonzero_torsions))
# print(min(np.abs(nonzero_torsions)))
# print(min(log_torsions))
log_curvatures_d_from_root = np.log10(nonzero_curvatures_d_from_root)
log_torsions_d_from_root = np.log10(nonzero_torsions_d_from_root)

log_slope_curvatures, log_intercept_curvatures, _, _, _ = scipy.stats.linregress(
    log_curvatures_d_from_root, log_curvatures
)
log_slope_curvatures = np.around(log_slope_curvatures, decimals=2)
log_intercept_curvatures = np.around(log_intercept_curvatures, decimals=2)
log_curvatures_fit = (
    log_slope_curvatures * log_curvatures_d_from_root + log_intercept_curvatures
)
curvatures_pearson_r, curvatures_p_value = scipy.stats.pearsonr(
    log_curvatures, log_curvatures_fit
)
print(
    log_slope_curvatures,
    log_intercept_curvatures,
    curvatures_pearson_r ** 2,
    curvatures_p_value,
)

log_slope_torsions, log_intercept_torsions, _, _, _ = scipy.stats.linregress(
    log_torsions_d_from_root, log_torsions
)
log_slope_torsions = np.around(log_slope_torsions, decimals=2)
log_intercept_torsions = np.around(log_intercept_torsions, decimals=2)
log_torsions_fit = (
    log_slope_torsions * log_torsions_d_from_root + log_intercept_torsions
)
torsions_pearson_r, torsions_p_value = scipy.stats.pearsonr(
    log_torsions, log_torsions_fit
)
print(
    log_slope_torsions,
    log_intercept_torsions,
    torsions_pearson_r ** 2,
    torsions_p_value,
)

fig = plt.figure(figsize=(22, 8))
axes = fig.subplots(1, 2)
GRAY = "#999999"
TITLE_TYPE_SETTINGS = {"fontname": "Arial", "size": 20}
SUP_TITLE_TYPE_SETTINGS = {"fontname": "Arial", "size": 24}
plt.rc("font", family="Arial", size=20)

ax = axes[0]
ax.spines["bottom"].set_color(GRAY)
ax.spines["top"].set_color(GRAY)
ax.spines["right"].set_color(GRAY)
ax.spines["left"].set_color(GRAY)
ax.tick_params(axis="both", colors=GRAY, labelsize="large")

ax.scatter(
    log_curvatures_d_from_root,
    log_curvatures,
    marker=".",
    label="Segment",
    color="#377eb8",
)
ax.plot(
    log_curvatures_d_from_root,
    log_curvatures_fit,
    color="#e41a1c",
    lw=2,
    label=r"$y={}x {}{}$".format(
        log_slope_curvatures,
        "+" if np.sign(log_intercept_curvatures) >= 0 else "-",
        np.abs(log_intercept_curvatures),
    ),
)
ax.set_title("Curvature")
ax.set_xlabel(r"$\log$ distance from segment root ($\mu m$)", fontsize=22)
ax.set_ylabel(r"$\log$ curvature", fontsize=22)
leg = ax.legend(loc=4)
leg.get_frame().set_edgecolor(GRAY)

ax = axes[1]
ax.spines["bottom"].set_color(GRAY)
ax.spines["top"].set_color(GRAY)
ax.spines["right"].set_color(GRAY)
ax.spines["left"].set_color(GRAY)
ax.tick_params(axis="both", colors=GRAY, labelsize="large")

ax.scatter(
    log_torsions_d_from_root,
    log_torsions,
    marker=".",
    label="Segment",
    color="#377eb8",
)
ax.plot(
    log_torsions_d_from_root,
    log_torsions_fit,
    color="#e41a1c",
    lw=2,
    label=r"$y={}x {}{}$".format(
        log_slope_torsions,
        "+" if np.sign(log_intercept_torsions) >= 0 else "-",
        np.abs(log_intercept_torsions),
    ),
)
ax.set_title("Torsion")
ax.set_xlabel(r"$\log$ distance from segment root ($\mu m$)", fontsize=22)
ax.set_ylabel(r"$\log$ absolute torsion", fontsize=22)
leg = ax.legend(loc=4)
leg.get_frame().set_edgecolor(GRAY)

fig.suptitle("Brain 2")
plt.savefig(os.path.join(experiment_dir, "{}_linear_regression_from_root.eps".format(brain)))
plt.savefig(os.path.join(experiment_dir, "{}_linear_regression_from_root.jpg".format(brain)))