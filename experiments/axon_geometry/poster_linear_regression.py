import numpy as np
import brainlit
import scipy
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

# EXTRACT DATA DIR FROM RELATIVE NOTEBOOK PATH
brain = "brain1"
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
mean_torsions = []
mean_curvatures = []
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
        swc_path = os.path.join(
            segments_swc_dir,
            "{}_G-{}_consensus.swc".format(
                "2018-08-01" if brain == "brain1" else "2018-12-01", string_id
            ),
        )
        df_swc_offset_neuron, _, _, _ = swc.read_swc_offset(swc_path)

        curvature_path = os.path.join(curvatures_dir, "{}.npy".format(i))
        torsion_path = os.path.join(torsions_dir, "{}.npy".format(i))

        print("Loaded segment {}".format(i))

        if os.path.exists(curvature_path) is True:
            curvature_data = np.load(curvature_path)
        if os.path.exists(torsion_path) is True:
            torsion_data = np.load(torsion_path)

        if (
            os.path.exists(curvature_path) is False
            or os.path.exists(torsion_path) is False
        ):
            G = GeometricGraph(df=df_swc_offset_neuron)
            print("Initialized GeometricGraph")
            spline_tree = G.fit_spline_tree_invariant()
            print("Computed splines")

            curvature_data = np.zeros((len(spline_tree.nodes), 2))
            torsion_data = np.zeros((len(spline_tree.nodes), 2))
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
                seg_length = u_um[-1] - u_um[0]
                curvature_data[j, 0] = seg_length
                torsion_data[j, 0] = seg_length
                # resample points at 1um
                uu = np.arange(u_um[0], u_um[-1] + 0.9, 1)
                # evaluate mean curvature of the segment
                _curvature = curvature(uu, t, c, k)
                mean_curvature = np.mean(_curvature)
                curvature_data[j, 1] = mean_curvature
                # evaluate mean torsion of the segment
                _torsion = torsion(uu, t, c, k)
                mean_torsion = np.mean(_torsion)
                torsion_data[j, 1] = mean_torsion

            np.save(curvature_path, curvature_data)
            np.save(torsion_path, torsion_data)

        seg_lengths.append(curvature_data[:, 0])
        mean_curvatures.append(curvature_data[:, 1])
        mean_torsions.append(torsion_data[:, 1])

seg_lengths = np.concatenate(seg_lengths)
mean_curvatures = np.concatenate(mean_curvatures)
print(max(mean_curvatures))
mean_torsions = np.concatenate(mean_torsions)


nonzero_mean_curvatures_masked = np.ma.masked_less(
    mean_curvatures, np.finfo(np.float64).eps
)
nonzero_mean_curvatures = nonzero_mean_curvatures_masked.compressed()
nonzero_mean_curvatures_seg_lengths = seg_lengths[
    nonzero_mean_curvatures_masked.mask == 0
]
nonzero_mean_torsions_masked = np.ma.masked_less(
    mean_torsions, np.finfo(np.float64).eps
)
nonzero_mean_torsions = nonzero_mean_torsions_masked.compressed()
nonzero_mean_torsions_seg_lengths = seg_lengths[nonzero_mean_torsions_masked.mask == 0]

log_mean_curvatures = np.log10(nonzero_mean_curvatures)
log_mean_torsions = np.log10(np.abs(nonzero_mean_torsions))
print(min(np.abs(nonzero_mean_torsions)))
print(min(log_mean_torsions))
log_curvatures_seg_lengths = np.log10(nonzero_mean_curvatures_seg_lengths)
log_torsions_seg_lengths = np.log10(nonzero_mean_torsions_seg_lengths)

log_slope_curvatures, log_intercept_curvatures, _, _, _ = scipy.stats.linregress(
    log_curvatures_seg_lengths, log_mean_curvatures
)
log_slope_curvatures = np.around(log_slope_curvatures, decimals=2)
log_intercept_curvatures = np.around(log_intercept_curvatures, decimals=2)
log_curvatures_fit = (
    log_slope_curvatures * log_curvatures_seg_lengths + log_intercept_curvatures
)
curvatures_pearson_r, curvatures_p_value = scipy.stats.pearsonr(
    log_mean_curvatures, log_curvatures_fit
)
print(
    log_slope_curvatures,
    log_intercept_curvatures,
    curvatures_pearson_r ** 2,
    curvatures_p_value,
)

log_slope_torsions, log_intercept_torsions, _, _, _ = scipy.stats.linregress(
    log_torsions_seg_lengths, log_mean_torsions
)
log_slope_torsions = np.around(log_slope_torsions, decimals=2)
log_intercept_torsions = np.around(log_intercept_torsions, decimals=2)
log_torsions_fit = (
    log_slope_torsions * log_torsions_seg_lengths + log_intercept_torsions
)
torsions_pearson_r, torsions_p_value = scipy.stats.pearsonr(
    log_mean_torsions, log_torsions_fit
)
print(
    log_slope_torsions,
    log_intercept_torsions,
    torsions_pearson_r ** 2,
    torsions_p_value,
)

fig = plt.figure(figsize=(15, 8))
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
    log_curvatures_seg_lengths,
    log_mean_curvatures,
    marker=".",
    label="Segment",
    color="#377eb8",
)
ax.plot(
    log_curvatures_seg_lengths,
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
ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=22)
ax.set_ylabel(r"$\log$ mean curvature", fontsize=22)
leg = ax.legend(loc=4)
leg.get_frame().set_edgecolor(GRAY)

ax = axes[1]
ax.spines["bottom"].set_color(GRAY)
ax.spines["top"].set_color(GRAY)
ax.spines["right"].set_color(GRAY)
ax.spines["left"].set_color(GRAY)
ax.tick_params(axis="both", colors=GRAY, labelsize="large")

ax.scatter(
    log_torsions_seg_lengths,
    log_mean_torsions,
    marker=".",
    label="Segment",
    color="#377eb8",
)
ax.plot(
    log_torsions_seg_lengths,
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
ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=22)
ax.set_ylabel(r"$\log$ mean absolute torsion", fontsize=22)
leg = ax.legend(loc=4)
leg.get_frame().set_edgecolor(GRAY)

fig.suptitle("Brain 1")

plt.savefig(os.path.join(experiment_dir, "{}_linear_regression.eps".format(brain)))
plt.savefig(os.path.join(experiment_dir, "{}_linear_regression.jpg".format(brain)))

# fig = plt.figure(figsize=(15, 6))
# axes = fig.subplots(1, 2)

# ax = axes[0]
# ax.spines["bottom"].set_color(GRAY)
# ax.spines["top"].set_color(GRAY)
# ax.spines["right"].set_color(GRAY)
# ax.spines["left"].set_color(GRAY)
# ax.tick_params(axis="both", colors=GRAY, labelsize="large")

# ax.scatter(
#     np.log10(seg_lengths), mean_curvatures, marker=".", label="Segment", color="#377eb8"
# )
# min_log_length = min(log_curvatures_seg_lengths)
# max_log_length = max(log_curvatures_seg_lengths)
# log_xx = np.linspace(min_log_length, max_log_length)
# ax.plot(
#     log_xx,
#     [
#         10 ** (log_intercept_curvatures + log_slope_curvatures * (u - min_log_length))
#         if u > min_log_length
#         else 0
#         for u in log_xx
#     ],
#     color="#e41a1c",
#     lw=2,
#     label=r"$10^{%.2fx %s%.2f}$"
#     % (
#         log_slope_curvatures,
#         "+" if np.sign(log_intercept_curvatures) >= 0 else "-",
#         np.abs(log_intercept_curvatures),
#     ),
# )
# ax.set_title("Curvature")
# ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=24)
# ax.set_ylabel(r"mean curvature", fontsize=24)
# ax.legend()

# ax = axes[1]
# ax.spines["bottom"].set_color(GRAY)
# ax.spines["top"].set_color(GRAY)
# ax.spines["right"].set_color(GRAY)
# ax.spines["left"].set_color(GRAY)
# ax.tick_params(axis="both", colors=GRAY, labelsize="large")

# ax.scatter(
#     np.log10(seg_lengths),
#     np.abs(mean_torsions),
#     marker=".",
#     label="Segment",
#     color="#377eb8",
# )
# min_log_length = min(log_torsions_seg_lengths)
# max_log_length = max(log_torsions_seg_lengths)
# log_xx = np.linspace(min_log_length, max_log_length)
# ax.plot(
#     log_xx,
#     [
#         10 ** (log_intercept_torsions + log_slope_torsions * (u - min_log_length))
#         if u > min_log_length
#         else 0
#         for u in log_xx
#     ],
#     color="#e41a1c",
#     lw=2,
#     label=r"$10^{%.2fx %s%.2f}$"
#     % (
#         log_slope_torsions,
#         "+" if np.sign(log_intercept_torsions) >= 0 else "-",
#         np.abs(log_intercept_torsions),
#     ),
# )
# ax.set_title("Torsion")
# ax.set_xlabel(r"$\log$ segment length ($\mu m$)", **TITLE_TYPE_SETTINGS)
# ax.set_ylabel(r"mean absolute torsion", **TITLE_TYPE_SETTINGS)
# ax.legend()

# fig.suptitle("Brain 2")

# plt.savefig(os.path.join(experiment_dir, "{}_results_semilog_s0.eps".format(brain)))
# plt.savefig(os.path.join(experiment_dir, "{}_results_semilog_s0.jpg".format(brain)))

# fig = plt.figure()
# log_seg_lengths = np.log10(seg_lengths)
# log_seg_lengths_within_mask = np.ma.masked_inside(log_seg_lengths, 1.75, 2.25)
# log_seg_lengths_within = log_seg_lengths_within_mask.compressed()
# mean_curvatures_within = mean_curvatures[log_seg_lengths_within_mask.mask == True]
# print(max(mean_curvatures_within))
# plt.hist(mean_curvatures_within, bins=50)
# plt.show()

# fig = plt.figure()
# log_seg_lengths = np.log10(seg_lengths)
# zero_curvatures_log_seg_lengths = log_seg_lengths[np.where(mean_curvatures > 1e-16)[0]]
# plt.hist(zero_curvatures_log_seg_lengths)
# plt.xlabel(r"$\log$ segment length ($\mu m$)", **TITLE_TYPE_SETTINGS)
# plt.ylabel("Number of segments (total = %d)" % len(seg_lengths))
# plt.title("curvature > 0")
# plt.show()

# min_log_length = min(log_seg_lengths)
# max_log_lengh = max(log_seg_lengths)
# bin_lims = np.linspace(min_log_length, max_log_length)
# l_bins = len(bin_lims) - 1
# r_zero_curvatures = []
# r_zero_torsions = []
# for i in np.arange(l_bins):
#     within_bin_log_lengths_mask = np.ma.masked_inside(
#         log_seg_lengths, bin_lims[i], bin_lims[i + 1]
#     )
#     # within_bin_log_lengths_mask = np.ma.masked_less_equal(log_seg_lengths, bin_lims[i+1])
#     within_bin_curvatures = mean_curvatures[within_bin_log_lengths_mask.mask == 1]
#     within_bin_torsions = np.abs(mean_torsions)[within_bin_log_lengths_mask.mask == 1]
#     zero_within_bin_curvatures = len(np.where(within_bin_curvatures < 1e-16)[0])
#     zero_within_bin_torsions = len(np.where(within_bin_torsions < 1e-16)[0])
#     r_zero_curvatures_bin = zero_within_bin_curvatures / len(within_bin_curvatures)
#     r_zero_torsions_bin = zero_within_bin_torsions / len(within_bin_torsions)
#     r_zero_curvatures.append(r_zero_curvatures_bin)
#     r_zero_torsions.append(r_zero_torsions_bin)

# print(len(np.where(mean_curvatures < 1e-16)[0])/len(mean_curvatures))
# print(len(np.where(np.abs(mean_torsions) < 1e-16)[0])/len(mean_torsions))

# fig = plt.figure()
# ax = fig.subplots(1, 1)
# ax.plot(bin_lims[:-1] + np.diff(bin_lims) / 2, r_zero_curvatures, label="curvatures")
# ax.plot(bin_lims[:-1] + np.diff(bin_lims) / 2, r_zero_torsions, label="torsions")
# ax.legend()
# ax.set_ylim([0, 1.05])
# ax.set_xlabel(r"$\log$ segment length ($\mu m$)", **TITLE_TYPE_SETTINGS)
# ax.set_ylabel(r"$P(y=0|X = x)$")
# plt.show()
