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
root_dir = data_dir = Path(__file__).parents[2]
data_dir = os.path.join(root_dir, "data/poster_reproduction")
experiment_dir = os.path.join(root_dir, "experiments/poster_reproduction")
segments_dir = os.path.join(data_dir, "segments")
segments_swc_dir = os.path.join(data_dir, "segments_swc")
splines_dir = os.path.join(data_dir, "splines")
curvatures_dir = os.path.join(data_dir, "curvatures")
torsions_dir = os.path.join(data_dir, "torsions")
print(segments_dir)

url_segments = "s3://open-neurodata/brainlit/brain1_segments"


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
            segments_swc_dir, "2018-08-01_G-{}_consensus.swc".format(string_id)
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
mean_torsions = np.concatenate(mean_torsions)

print(np.where(np.isnan(mean_torsions) == 0))

nonzero_mean_curvatures_masked = np.ma.masked_equal(mean_curvatures, 0)
nonzero_mean_curvatures = nonzero_mean_curvatures_masked.compressed()
nonzero_mean_curvatures_seg_lengths = seg_lengths[
    nonzero_mean_curvatures_masked.mask == 0
]
nonzero_mean_torsions_masked = np.ma.masked_equal(mean_torsions, 0)
nonzero_mean_torsions = nonzero_mean_torsions_masked.compressed()
nonzero_mean_torsions_seg_lengths = seg_lengths[nonzero_mean_torsions_masked.mask == 0]

log_mean_curvatures = np.log(nonzero_mean_curvatures)
log_mean_torsions = np.log(np.abs(nonzero_mean_torsions))
log_curvatures_seg_lengths = np.log(nonzero_mean_curvatures_seg_lengths)
log_torsions_seg_lengths = np.log(nonzero_mean_torsions_seg_lengths)

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

fig = plt.figure(figsize=(12, 8))
axes = fig.subplots(1, 2)

# ax = axes[0, 0]
# ax.scatter(np.log(np.concatenate(seg_lengths)), np.log(np.concatenate(mean_curvatures)))
# ax.set_title("Curvature")
# ax.set_xlabel(r"$\log_{e}$ segment length ($\mu m$)")
# ax.set_ylabel(r"$\log_{e}$ mean curvature")

# ax = axes[0, 1]
# ax.scatter(np.log(np.concatenate(seg_lengths)), np.log(np.concatenate(mean_torsions)))
# ax.set_title("Torsion")
# ax.set_xlabel(r"$\log_{e}$ segment length ($\mu m$)")
# ax.set_ylabel(r"$\log_{e}$ mean torsion")

ax = axes[0]
ax.scatter(
    log_curvatures_seg_lengths, log_mean_curvatures, label="Segment mean curvature"
)
ax.plot(
    log_curvatures_seg_lengths,
    log_curvatures_fit,
    "r",
    label=r"$y={}x {}{}, r^2={}$".format(
        log_slope_curvatures,
        "+" if np.sign(log_intercept_curvatures) >= 0 else "-",
        np.abs(log_intercept_curvatures),
        np.around(curvatures_pearson_r ** 2, decimals=2),
    ),
)
ax.set_title("Curvature")
ax.set_xlabel(r"$\log$ segment length ($\mu m$)")
ax.set_ylabel(r"$\log$ mean curvature")
ax.legend()

ax = axes[1]
ax.scatter(log_torsions_seg_lengths, log_mean_torsions, label="Segment mean torsion")
ax.plot(
    log_torsions_seg_lengths,
    log_torsions_fit,
    "r",
    label=r"$y={}x {}{}, r^2={}$".format(
        log_slope_torsions,
        "+" if np.sign(log_intercept_torsions) >= 0 else "-",
        np.abs(log_intercept_torsions),
        np.around(torsions_pearson_r ** 2, decimals=2),
    ),
)
ax.set_title("Torsion")
ax.set_xlabel(r"$\log$ segment length ($\mu m$)")
ax.set_ylabel(r"$\log$ mean absolute torsion")
ax.legend()

plt.savefig(os.path.join(experiment_dir, "results.eps"))
plt.savefig(os.path.join(experiment_dir, "results.jpg"))