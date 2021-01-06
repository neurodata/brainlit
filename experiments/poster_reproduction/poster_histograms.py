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
brain = "brain1"
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
        swc_path = os.path.join(
            segments_swc_dir,
            "{}_G-{}_consensus.swc".format(
                "2018-08-01" if brain == "brain1" else "2018-12-01", string_id
            ),
        )
        df_swc_offset_neuron, _, _, _ = swc.read_swc_offset(swc_path)

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

# f = lambda x, a, k, u: [a*xx**(-k) if xx > u else 0 for xx in x]
# (a, k, u), _ = scipy.optimize.curve_fit(f, seg_lengths, mean_curvatures, bounds=([0, 0, 5], [np.inf, np.inf, 15]))
# print(a, k, u)
# slope_curvatures, intercept_curvatures, _, _, _ = scipy.stats.linregress(
#     seg_lengths, mean_curvatures
# )
# print(slope_curvatures, intercept_curvatures)
# # a, loc, scale = scipy.stats.powerlaw.fit(mean_curvatures)
# # print(a, loc, scale)
# xx = np.linspace(1, int(1e4), 3000)
# # plt.plot(xx, scipy.stats.powerlaw(a, loc, scale).pdf(xx))
# plt.plot(xx, f(xx, a, k, u), xx, xx*slope_curvatures + intercept_curvatures)
# plt.scatter(seg_lengths, mean_curvatures, color="g", marker=".", alpha=0.2)
# plt.ylim([-0.5, 2])
# # plt.xscale("log")
# plt.show()

# fig = plt.figure(figsize=(21, 7))
# axes = fig.subplots(1, 2)
# GRAY = "#999999"
# TITLE_TYPE_SETTINGS = {"fontname": "Arial", "size": 20}
# SUP_TITLE_TYPE_SETTINGS = {"fontname": "Arial", "size": 24}
# plt.rc("font", family="Arial", size=20)

# log_seg_lengths = np.log10(seg_lengths)
# min_log_seg_length = min(log_seg_lengths)
# max_log_seg_length = max(log_seg_lengths)
# xx = np.linspace(min_log_seg_length, max_log_seg_length, 1000)[:, np.newaxis]

# ax = axes[0]
# ax.spines["bottom"].set_color(GRAY)
# ax.spines["top"].set_color(GRAY)
# ax.spines["right"].set_color(GRAY)
# ax.spines["left"].set_color(GRAY)
# ax.tick_params(axis="both", colors=GRAY, labelsize="large")

# zero_curvatures_log_seg_lengths = log_seg_lengths[np.where(mean_curvatures < 1e-16)[0]]
# nonzero_curvatures_log_seg_lengths = log_seg_lengths[
#     np.where(mean_curvatures > 1e-16)[0]
# ]
# zero_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
#     zero_curvatures_log_seg_lengths[:, np.newaxis]
# )
# nonzero_kde = KernelDensity(kernel="gaussian", bandwidth=0.25).fit(
#     nonzero_curvatures_log_seg_lengths[:, np.newaxis]
# )
# zero_log_dens = zero_kde.score_samples(xx)
# nonzero_log_dens = nonzero_kde.score_samples(xx)
# # ax.hist(zero_curvatures_log_seg_lengths, density=True)
# # ax.hist(nonzero_curvatures_log_seg_lengths, density=True)
# alpha_zero_curvatures = len(zero_curvatures_log_seg_lengths) / len(seg_lengths)
# alpha_nonzero_curvatures = len(nonzero_curvatures_log_seg_lengths) / len(seg_lengths)
# print(alpha_zero_curvatures, alpha_nonzero_curvatures)
# zero_norm_pdf = alpha_zero_curvatures * np.exp(zero_log_dens)
# nonzero_norm_pdf = alpha_nonzero_curvatures * np.exp(nonzero_log_dens)

# # ax.plot(xx.squeeze(), zero_norm_pdf, label=r"$c=0$")
# ax.fill_between(xx.squeeze(), 0, zero_norm_pdf, alpha=0.7, label=r"$\mathcal{k} = 0$")
# # ax.plot(xx.squeeze(), nonzero_norm_pdf, label=r"$c=0$")
# ax.fill_between(
#     xx.squeeze(), 0, nonzero_norm_pdf, alpha=0.7, label=r"$\mathcal{k} > 0$"
# )

# mask = np.array(
#     [
#         False if zero_ > nonzero_ else True
#         for zero_, nonzero_ in zip(zero_norm_pdf, nonzero_norm_pdf)
#     ]
# )
# ids = np.where(mask == True)[0]
# xx_dashed = xx.squeeze()[ids]
# zero_norm_pdf_dashed = zero_norm_pdf[ids]
# ax.plot(xx_dashed.squeeze(), zero_norm_pdf_dashed, "--")


# ax.set_title(r"Curvature ($\alpha = %.2f$)" % alpha_zero_curvatures)
# ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=24)
# ax.set_ylabel(r"pdf", fontsize=24)
# leg = ax.legend(loc=1)
# leg.get_frame().set_edgecolor(GRAY)
# ax.set_xticks([1, 2, 3, 4])

# ax = axes[1]
# ax.spines["bottom"].set_color(GRAY)
# ax.spines["top"].set_color(GRAY)
# ax.spines["right"].set_color(GRAY)
# ax.spines["left"].set_color(GRAY)
# ax.tick_params(axis="both", colors=GRAY, labelsize="large")

# zero_torsions_log_seg_lengths = log_seg_lengths[np.where(mean_torsions < 1e-16)[0]]
# nonzero_torsions_log_seg_lengths = log_seg_lengths[np.where(mean_torsions > 1e-16)[0]]
# zero_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
#     zero_torsions_log_seg_lengths[:, np.newaxis]
# )
# nonzero_kde = KernelDensity(kernel="gaussian", bandwidth=0.25).fit(
#     nonzero_torsions_log_seg_lengths[:, np.newaxis]
# )
# zero_log_dens = zero_kde.score_samples(xx)
# nonzero_log_dens = nonzero_kde.score_samples(xx)
# # ax.hist(zero_torsions_log_seg_lengths, density=True)
# # ax.hist(nonzero_torsions_log_seg_lengths, density=True)
# alpha_zero_torsions = len(zero_torsions_log_seg_lengths) / len(seg_lengths)
# alpha_nonzero_torsions = len(nonzero_torsions_log_seg_lengths) / len(seg_lengths)
# print(alpha_zero_torsions, alpha_nonzero_torsions)
# zero_norm_pdf = alpha_zero_torsions * np.exp(zero_log_dens)
# nonzero_norm_pdf = alpha_nonzero_torsions * np.exp(nonzero_log_dens)
# # ax.plot(xx.squeeze(), zero_norm_pdf, label=r"$c=0$")
# ax.fill_between(xx.squeeze(), 0, zero_norm_pdf, alpha=0.7, label=r"$\tau = 0$")
# # ax.plot(xx.squeeze(), nonzero_norm_pdf, label=r"$c=0$")
# ax.fill_between(xx.squeeze(), 0, nonzero_norm_pdf, alpha=0.7, label=r"$\tau > 0$")

# mask = np.array(
#     [
#         False if zero_ > nonzero_ else True
#         for zero_, nonzero_ in zip(zero_norm_pdf, nonzero_norm_pdf)
#     ]
# )
# ids = np.where(mask == True)[0]
# xx_dashed = xx.squeeze()[ids]
# zero_norm_pdf_dashed = zero_norm_pdf[ids]
# ax.plot(xx_dashed.squeeze(), zero_norm_pdf_dashed, "--")

# ax.set_title(r"Torsion ($\alpha = %.2f$)" % alpha_zero_torsions)
# ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=24)
# ax.set_ylabel(r"pdf", fontsize=24)
# leg = ax.legend(loc=1)
# leg.get_frame().set_edgecolor(GRAY)
# ax.set_xticks([1, 2, 3, 4])


# fig.suptitle("Brain 1")
# plt.savefig(os.path.join(experiment_dir, "{}_histograms.jpg".format(brain)))
# plt.savefig(os.path.join(experiment_dir, "{}_histograms.eps".format(brain)))
# plt.show()

###################################################################################

fig = plt.figure(figsize=(21, 7))
axes = fig.subplots(1, 2)
GRAY = "#999999"
TITLE_TYPE_SETTINGS = {"fontname": "Arial", "size": 20}
SUP_TITLE_TYPE_SETTINGS = {"fontname": "Arial", "size": 24}
plt.rc("font", family="Arial", size=20)

# log_seg_lengths = np.log10(seg_lengths)
log_d_from_root = np.log10(d_from_root)
min_log_d_from_root = min(log_d_from_root)
max_log_d_from_root = max(log_d_from_root)
xx = np.linspace(min_log_d_from_root, max_log_d_from_root, 1000)[:, np.newaxis]

ax = axes[0]
ax.spines["bottom"].set_color(GRAY)
ax.spines["top"].set_color(GRAY)
ax.spines["right"].set_color(GRAY)
ax.spines["left"].set_color(GRAY)
ax.tick_params(axis="both", colors=GRAY, labelsize="large")

zero_curvatures_log_d_from_root = log_d_from_root[np.where(curvatures < 1e-16)[0]]
nonzero_curvatures_log_d_from_root = log_d_from_root[np.where(curvatures > 1e-16)[0]]
torch_nonzero_curvatures_log_d_from_root = torch.from_numpy(
    nonzero_curvatures_log_d_from_root
).to(device)
bins = 100
# compute histogram of zero-curvatures
zero_curvatures_hist_min = np.min(zero_curvatures_log_d_from_root)
zero_curvatures_hist_max = np.max(zero_curvatures_log_d_from_root)
zero_curvatures_hist_bin_edges = np.arange(min, max, bins)
torch_zero_curvatures_log_d_from_root = torch.from_numpy(
    zero_curvatures_log_d_from_root
)
zero_curvature_hist = torch.histc(
    torch_zero_curvatures_log_d_from_root,
    bins=100,
    min=zero_curvatures_hist_min,
    max=zero_curvatures_hist_max,
)
# compute histogram of non-zero-curvatures (uses GPU)
nonzero_curvatures_hist = torch.histc(torch_nonzero_curvatures_log_d_from_root).to(
    device
)
nonzero_curvatures_hist_min = torch.min(torch_nonzero_curvatures_log_d_from_root)
nonzero_curvatures_hist_max = torch.max(torch_nonzero_curvatures_log_d_from_root)
nonzero_curvatures_hist_bin_edges = torch.arange(
    nonzero_curvatures_hist_min,
    nonzero_curvatures_hist_max,
    (nonzero_curvatures_hist_max - nonzero_curvatures_hist_min) / bins,
)
nonzero_curvatures_hist = torch.histc(
    torch_nonzero_curvatures_log_d_from_root,
    bins=bins,
    min=nonzero_curvatures_hist_min,
    max=nonzero_curvatures_hist_max
)
zero_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
    zero_curvatures_log_d_from_root[:, np.newaxis]
)
# print("done KDE zero curvatures")
# nonzero_kde = KernelDensity(kernel="gaussian", bandwidth=0.25).fit(
#     nonzero_curvatures_log_d_from_root[:, np.newaxis]
# )
# print("done KDE non-zero curvatures")
# zero_log_dens = zero_kde.score_samples(xx)
# nonzero_log_dens = nonzero_kde.score_samples(xx)
# ax.hist(zero_curvatures_log_seg_lengths, density=True)
# ax.hist(nonzero_curvatures_log_seg_lengths, density=True)
alpha_zero_curvatures = len(zero_curvatures_log_d_from_root) / len(d_from_root)
alpha_nonzero_curvatures = len(nonzero_curvatures_log_d_from_root) / len(d_from_root)
print(alpha_zero_curvatures, alpha_nonzero_curvatures)
# zero_norm_pdf = alpha_zero_curvatures * np.exp(zero_log_dens)
# nonzero_norm_pdf = alpha_nonzero_curvatures * np.exp(nonzero_log_dens)

# # ax.plot(xx.squeeze(), zero_norm_pdf, label=r"$c=0$")
# ax.fill_between(xx.squeeze(), 0, zero_norm_pdf, alpha=0.7, label=r"$\mathcal{k} = 0$")
# # ax.plot(xx.squeeze(), nonzero_norm_pdf, label=r"$c=0$")
# ax.fill_between(
#     xx.squeeze(), 0, nonzero_norm_pdf, alpha=0.7, label=r"$\mathcal{k} > 0$"
# )
ax.bar(zero_curvatures_hist_bin_edges, zero_curvature_hist.cpu().numpy())
ax.bar(nonzero_curvatures_hist_bin_edges, nonzero_curvatures_hist.cpu().numpy())

# mask = np.array(
#     [
#         False if zero_ > nonzero_ else True
#         for zero_, nonzero_ in zip(zero_norm_pdf, nonzero_norm_pdf)
#     ]
# )
# ids = np.where(mask == True)[0]
# xx_dashed = xx.squeeze()[ids]
# zero_norm_pdf_dashed = zero_norm_pdf[ids]
# ax.plot(xx_dashed.squeeze(), zero_norm_pdf_dashed, "--")


ax.set_title(r"Curvature ($\alpha = %.2f$)" % alpha_zero_curvatures)
ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=24)
ax.set_ylabel(r"pdf", fontsize=24)
leg = ax.legend(loc=1)
leg.get_frame().set_edgecolor(GRAY)
# ax.set_xticks([1, 2, 3, 4])

ax = axes[1]
ax.spines["bottom"].set_color(GRAY)
ax.spines["top"].set_color(GRAY)
ax.spines["right"].set_color(GRAY)
ax.spines["left"].set_color(GRAY)
ax.tick_params(axis="both", colors=GRAY, labelsize="large")

zero_torsions_log_d_from_root = d_from_root[np.where(torsions < 1e-16)[0]]
nonzero_torsions_log_d_from_root = d_from_root[np.where(torsions > 1e-16)[0]]
# zero_kde = KernelDensity(kernel="gaussian", bandwidth=0.1).fit(
#     zero_torsions_log_d_from_root[:, np.newaxis]
# )
# nonzero_kde = KernelDensity(kernel="gaussian", bandwidth=0.25).fit(
#     nonzero_torsions_log_d_from_root[:, np.newaxis]
# )
# zero_log_dens = zero_kde.score_samples(xx)
# nonzero_log_dens = nonzero_kde.score_samples(xx)
# # ax.hist(zero_torsions_log_seg_lengths, density=True)
# # ax.hist(nonzero_torsions_log_seg_lengths, density=True)
# alpha_zero_torsions = len(zero_torsions_log_d_from_root) / len(d_from_root)
# alpha_nonzero_torsions = len(nonzero_torsions_log_d_from_root) / len(d_from_root)
# print(alpha_zero_torsions, alpha_nonzero_torsions)
# zero_norm_pdf = alpha_zero_torsions * np.exp(zero_log_dens)
# nonzero_norm_pdf = alpha_nonzero_torsions * np.exp(nonzero_log_dens)
# # ax.plot(xx.squeeze(), zero_norm_pdf, label=r"$c=0$")
# ax.fill_between(xx.squeeze(), 0, zero_norm_pdf, alpha=0.7, label=r"$\tau = 0$")
# # ax.plot(xx.squeeze(), nonzero_norm_pdf, label=r"$c=0$")
# ax.fill_between(xx.squeeze(), 0, nonzero_norm_pdf, alpha=0.7, label=r"$\tau > 0$")

# mask = np.array(
#     [
#         False if zero_ > nonzero_ else True
#         for zero_, nonzero_ in zip(zero_norm_pdf, nonzero_norm_pdf)
#     ]
# )
# ids = np.where(mask == True)[0]
# xx_dashed = xx.squeeze()[ids]
# zero_norm_pdf_dashed = zero_norm_pdf[ids]
# ax.plot(xx_dashed.squeeze(), zero_norm_pdf_dashed, "--")

# ax.set_title(r"Torsion ($\alpha = %.2f$)" % alpha_zero_torsions)
# ax.set_xlabel(r"$\log$ segment length ($\mu m$)", fontsize=24)
# ax.set_ylabel(r"pdf", fontsize=24)
# leg = ax.legend(loc=1)
# leg.get_frame().set_edgecolor(GRAY)
# ax.set_xticks([1, 2, 3, 4])

fig.suptitle("Brain 1")
plt.savefig(os.path.join(experiment_dir, "{}_histograms_from_root.jpg".format(brain)))
plt.savefig(os.path.join(experiment_dir, "{}_histograms_from_root.eps".format(brain)))
plt.show()