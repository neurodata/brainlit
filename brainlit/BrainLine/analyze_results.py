import numpy as np
from cloudreg.scripts.transform_points import NGLink
from cloudvolume import CloudVolume, exceptions
from tqdm import tqdm
from skimage import io, measure
from brainlit.BrainLine.util import (
    _find_atlas_level_label,
    _fold,
    _setup_atlas_graph,
    _get_atlas_level_nodes,
    _get_corners,
)
import napari
import scipy.ndimage as ndi
from scipy.stats import ttest_ind
from scipy.signal import convolve2d
import seaborn as sns
from statannotations.Annotator import Annotator
from statannotations.stats.StatTest import StatTest
from statannotations import stats
import pandas as pd
import matplotlib.pyplot as plt
import os
from joblib import Parallel, delayed
import pickle
from brainrender import Scene
from brainrender.actors import Points, Volume
import json
from cloudvolume.exceptions import OutOfBoundsError


class BrainDistribution:
    """Helps generate plots and visualizations for brain connection distributions across subtyeps.

    Arguments:
        brain_ids (list): List of brain IDs (keys of data json file).

    Attributes:
        brain_ids (list): List of brain IDs (keys of data json file).
    """

    def __init__(self, brain_ids: list, data_file: str):
        self.brain_ids = brain_ids
        with open(data_file) as f:
            data = json.load(f)
        self.brain2paths = data["brain2paths"]
        self.subtype_counts = self._get_subtype_counts()

    def _slicetolabels(self, slice, fold_on: bool = False, atlas_level: int = 5):
        region_graph = _setup_atlas_graph()
        atlas_level_nodes = _get_atlas_level_nodes(atlas_level, region_graph)
        newslice = np.copy(slice)
        new_labels = {}

        for label in tqdm(np.unique(slice), desc=f"Relabeling slice"):
            atlas_level_label = _find_atlas_level_label(
                label, atlas_level_nodes, atlas_level, region_graph
            )
            newslice[slice == label] = atlas_level_label
            if atlas_level_label not in new_labels.keys():
                if atlas_level_label in region_graph.nodes:
                    name = region_graph.nodes[atlas_level_label]["name"]
                else:
                    name = "??"
                new_labels[atlas_level_label] = name

        labels = measure.label(newslice)
        borders = 0 * labels
        for label in tqdm(np.unique(labels), desc=f"Processing labels"):
            if label != 0:
                mask = np.array(labels == label, dtype="int")
                erode = np.array(ndi.binary_erosion(mask))
                outline = mask - erode
                borders += outline

        if fold_on:
            half_width = np.round(borders.shape[1] / 2).astype(int)
            borders = borders[:, :half_width]
            newslice = newslice[:, :half_width]
        else:
            half_width = -1
        return newslice, borders, half_width

    def _get_subtype_counts(self):
        brain2paths = self.brain2paths
        brain_ids = self.brain_ids
        counts = {}
        for brain_id in brain_ids:
            subtype = brain2paths[brain_id]["subtype"]
            if subtype in counts.keys():
                counts[subtype] = counts[subtype] + 1
            else:
                counts[subtype] = 1
        return counts


class SomaDistribution(BrainDistribution):
    """Object to generate various analysis images for results from a set of brain IDs. An implementation of BrainDistribution class.

    Arguments:
        brain_ids (list): List of brain IDs (keys of data json file).
        data_files (str): Path to json file that contains information about samples.
        show_plots (bool): Whether to show plots, only works if you have graphics access.

    Attributes:
        brain2paths (dict): Information about different data samples.
        show_plots (bool): Whether to show plots, only works if you have graphics access.
        atlas_points (dict): Key - sample ID, Value - coordinates of soma detection.
        id_to_regioncounts (dict): Key - sample ID, Value - dictionary of detection counts by region.
        region_graph (nx.DiGraph): Graph of region hierarchy with soma detection counts as node attributes.

    """

    def __init__(self, brain_ids: list, data_file: str, show_plots: bool = True):
        super().__init__(brain_ids, data_file)
        self.show_plots = show_plots

        atlas_points = self._retrieve_soma_coords(brain_ids)
        self.atlas_points = atlas_points
        id_to_regioncounts = self._get_regions(atlas_points)
        self.id_to_regioncounts = id_to_regioncounts
        region_graph = self._setup_regiongraph()
        self.region_graph = region_graph

    def _retrieve_soma_coords(self, brain_ids: list):
        brain2paths = self.brain2paths
        atlas_points = {}
        for brain_id in brain_ids:
            if "somas_atlas_path" in brain2paths[brain_id].keys():
                json_dir = brain2paths[brain_id]["somas_atlas_path"]
                jsons = os.listdir(json_dir)
                points = []
                for json_file in jsons:
                    json_path = json_dir + json_file
                    print(
                        f"Brain {brain_id}: Collecting atlas space soma points from file: {json_path}"
                    )
                    with open(json_path) as f:
                        data = json.load(f)
                    for pt in data:
                        points.append(pt["point"])
                atlas_points[brain_id] = np.array(points)
            elif "somas_atlas_url" in brain2paths[brain_id].keys():
                viz_link = brain2paths[brain_id]["somas_atlas_url"]
                viz_link = NGLink(viz_link.split("json_url=")[-1])
                ngl_json = viz_link._json
                for layer in ngl_json["layers"]:
                    if layer["type"] == "annotation":
                        points = []
                        for annot in layer["annotations"]:
                            points.append(annot["point"])

                        print(
                            f'Brain {brain_id}: Collecting atlas space soma points from layer: {layer["name"]}'
                        )
                atlas_points[brain_id] = np.array(points)
            else:
                raise ValueError(f"No somas_atlas_url layer for brain: {brain_id}")

        return atlas_points

    def _get_regions(self, points: dict):
        brain2paths = self.brain2paths
        if "filepath" in brain2paths["atlas"].keys():
            vol_atlas = io.imread(brain2paths["atlas"]["filepath"])
        else:
            vol_atlas = CloudVolume(brain2paths["atlas"]["url"])

        id_to_regioncounts = {}
        for brain_id in tqdm(points.keys(), desc="Finding soma regions of brains"):
            brain_dict = {}
            for point in tqdm(
                points[brain_id], desc="Finding soma regions", leave=False
            ):
                point_int = [int(np.round(p)) for p in point]
                try:
                    region = int(vol_atlas[point_int[0], point_int[1], point_int[2]])
                except IndexError:
                    continue
                except OutOfBoundsError:
                    continue
                if region in brain_dict.keys():
                    brain_dict[region] = brain_dict[region] + 1
                else:
                    brain_dict[region] = 1

            id_to_regioncounts[brain_id] = brain_dict

        return id_to_regioncounts

    def napari_coronal_section(
        self,
        z: int,
        subtype_colors: dict,
        symbols: list = ["o", "+", "^", "vbar"],
        fold_on: bool = False,
    ):
        """Generate napari view with allen atlas and points of soma detections.

        Args:
            z (int): index of coronal slice in Allen atlas.
            subtype_colors (dict): Mapping of subtypes (in soma_data.py file) to colors for soma plotting.
            symbols (list): Napari point symbols to use for different samples of the same subtype. Defaults to ["o", "+", "^", "vbar"].
            fold_on (bool, optional): Whether napari views should be a hemisphere, in which case detections from the other side are mirrored. Defaults to False.
        """
        brain2paths = self.brain2paths
        atlas_points = self.atlas_points
        subtype_counts = self._get_subtype_counts()

        if "filepath" in brain2paths["atlas"].keys():
            vol_atlas = io.imread(brain2paths["atlas"]["filepath"])
        else:
            vol_atlas = CloudVolume(brain2paths["atlas"]["url"])

        slice = np.squeeze(vol_atlas[z, :, :])
        newslice, borders, half_width = self._slicetolabels(slice, fold_on=fold_on)
        heatmap = np.zeros([borders.shape[0], borders.shape[1], 3])

        if self.show_plots:
            v = napari.Viewer()
            v.add_labels(newslice, scale=[10, 10])

        gtype_counts = {}
        for key in subtype_colors.keys():
            gtype_counts[key] = 0

        for key in tqdm(atlas_points.keys(), desc=f"Processing brains in z={z}"):
            ra = atlas_points[key]
            gtype = brain2paths[key]["subtype"]

            ra = atlas_points[key]
            points = ra[(ra[:, 0] < z + 10) & (ra[:, 0] > z - 10)]

            # only select points that fall on an ROI
            fg_points = []
            channel_map = {"red": 0, "green": 1, "blue": 2}
            channel = channel_map[subtype_colors[gtype]]
            for point in points:
                im_coord = [int(point[1]), int(point[2])]

                if (
                    im_coord[0] in range(0, newslice.shape[0])
                    and im_coord[1] in range(0, newslice.shape[1])
                    and newslice[im_coord[0], im_coord[1]] != 0
                ):
                    if fold_on and im_coord[1] >= half_width:
                        im_coord[1] = 2 * half_width - im_coord[1]

                    fg_points.append([im_coord[0], im_coord[1]])
                    heatmap[
                        int(im_coord[0]), int(im_coord[1]), channel
                    ] += 1  # /subtype_counts[gtype]
            if self.show_plots:
                v.add_points(
                    fg_points,
                    symbol=symbols[gtype_counts[gtype]],
                    face_color=subtype_colors[gtype],
                    size=10,
                    name=f"{key}: {gtype}",
                    scale=[10, 10],
                )

            gtype_counts[gtype] = gtype_counts[gtype] + 1

        if self.show_plots:
            for c in range(3):
                heatmap[:, :, c] = ndi.gaussian_filter(heatmap[:, :, c], sigma=3)
            v.add_image(heatmap, scale=[10, 10], name=f"Heatmap")  # , rgb=True)
            v.add_labels(borders * 2, scale=[10, 10], name=f"z={z}")

            v.scale_bar.unit = "um"
            v.scale_bar.visible = True
            napari.run()

    def brainrender_somas(self, subtype_colors: dict, brain_region: str = "DR"):
        """Generate brainrender viewer with soma detections.

        Args:
            subtype_colors (_type_): Mapping of subtypes (in soma_data.py file) to colors for soma plotting.
            brain_region (str, optional): Brain region to display with the detections (e.g. dorsal raphe nucleus). Defaults to "DR".
        """
        brain_ids = self.brain_ids
        brain2paths = self.brain2paths

        scene = Scene(atlas_name="allen_mouse_50um", title="Input Somas")
        scene.add_brain_region(brain_region, alpha=0.15)

        for brain_id in brain_ids:
            viz_link = brain2paths[brain_id]["somas_atlas_url"]
            viz_link = NGLink(viz_link.split("json_url=")[-1])
            ngl_json = viz_link._json
            for layer in ngl_json["layers"]:
                if layer["type"] == "annotation":
                    points = []
                    for annot in tqdm(
                        layer["annotations"],
                        desc="Going through points...",
                        leave=False,
                    ):
                        struct_coord = np.array(annot["point"]) / 5
                        try:
                            region = scene.atlas.structure_from_coords(struct_coord)
                            if region != 0 and np.any(struct_coord < 0) == False:
                                points.append(annot["point"])
                        except IndexError:
                            continue
                    points = np.array(points) * 10

            scene.add(
                Points(
                    points,
                    name="CELLS",
                    colors=subtype_colors[brain2paths[brain_id]["subtype"]],
                )
            )

        scene.content

        if self.show_plots:
            scene.render()

    def region_barchart(
        self, regions: list, composite_regions: dict = {}, normalize_region: int = -1
    ):
        """Generate bar charts comparing soma detection counts between regions.

        Args:
            regions (list): List of Allen atlas brain region IDs to display data for (ID's found here: http://api.brain-map.org/api/v2/structure_graph_download/1.json)
            composite_regions (dict, optional): Mapping from a custom composite region (str, e.g. "Amygdala") to a set of regions that compose it (list of ints e.g. [131, 295, 319, 780]). Defaults to {}.
            normalize_region (int, optional): Region ID to normalize data for the normalized bar chart. Defaults to -1.
        """
        subtype_counts = self._get_subtype_counts()
        id_to_somatotals = self._count_somas()

        df = self._make_bar_df(
            regions,
            composite_regions,
            id_to_somatotals,
            subtype_counts,
            normalize_region,
        )
        subtypes = df["Subtype"].unique()

        if normalize_region >= 0:
            fig, axes = plt.subplots(1, 3, figsize=(39, 20))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(26, 20))

        sns.set(font_scale=2)

        # first panel
        fig_args = {
            "x": "Somas (#)",
            "y": "Region",
            "hue": "Subtype",
            "data": df,
        }

        sns.set(font_scale=2)
        bplot = sns.barplot(ax=axes[0], orient="h", **fig_args)
        bplot.set_xscale("log")

        if len(subtypes) > 1:
            annotator = self._configure_annotator(df, axes[0], "Somas (#)")
            annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
            annotator.apply_and_annotate()

        # second panel
        fig_args = {
            "x": "Percent of Total Somas (%)",
            "y": "Region",
            "hue": "Subtype",
            "data": df,
        }

        bplot = sns.barplot(ax=axes[1], orient="h", **fig_args)
        bplot.set_xscale("log")

        if len(subtypes) > 1:
            annotator = self._configure_annotator(
                df, axes[1], "Percent of Total Somas (%)"
            )
            annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
            annotator.apply_and_annotate()

        # third panel
        if normalize_region >= 0:
            fig_args = {
                "x": "Normalized Somas",
                "y": "Region",
                "hue": "Subtype",
                "data": df,
            }

            sns.set(font_scale=2)
            bplot = sns.barplot(ax=axes[2], orient="h", **fig_args)
            bplot.set_xscale("log")

            if len(subtypes) > 1:
                annotator = self._configure_annotator(df, axes[2], "Normalized Somas")
                annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
                annotator.apply_and_annotate()

        fig.tight_layout()
        if self.show_plots:
            plt.show()

    def _setup_regiongraph(self):
        brain_ids = self.brain_ids
        id_to_regioncounts = self.id_to_regioncounts
        region_graph = _setup_atlas_graph()
        max_level = 0

        # set to 0
        for node in region_graph.nodes:
            if region_graph.nodes[node]["level"] > max_level:
                max_level = region_graph.nodes[node]["level"]
            for brain_id in brain_ids:
                region_graph.nodes[node][brain_id] = 0

        # add counts
        for brain_id in brain_ids:
            regioncounts = id_to_regioncounts[brain_id]
            for region in regioncounts.keys():
                if region in region_graph.nodes:
                    region_graph.nodes[region][brain_id] = (
                        region_graph.nodes[region][brain_id]
                        + id_to_regioncounts[brain_id][region]
                    )

        # propagate counts up the hierarchy
        for brain_id in brain_ids:
            for lvl in range(max_level, 0, -1):
                for node in region_graph.nodes:
                    if region_graph.nodes[node]["level"] == lvl:
                        parent = list(region_graph.in_edges(node))[0][0]
                        region_graph.nodes[parent][brain_id] = (
                            region_graph.nodes[parent][brain_id]
                            + region_graph.nodes[node][brain_id]
                        )

        return region_graph

    def _count_somas(self):
        id_to_somatotals = {}
        brain_ids = self.brain_ids
        atlas_points = self.atlas_points

        for brain_id in brain_ids:
            points = atlas_points[brain_id]
            id_to_somatotals[brain_id] = points.shape[0]
        return id_to_somatotals

    def _make_bar_df(
        self,
        regions: list,
        composite_regions,
        id_to_somatotals: dict,
        subtype_counts: dict,
        normalize_region,
    ):
        region_graph = self.region_graph
        brain_ids = self.brain_ids

        subtypes = []
        somas = []
        somas_norm = []
        somas_pct = []
        region_name = []
        brain_ids_data = []

        for region in regions:
            print(f"Populating: {region_graph.nodes[region]['name']}")
            for brain_id in brain_ids:
                subtype = self.brain2paths[brain_id]["subtype"]
                soma_count = region_graph.nodes[region][brain_id]
                somas.append(soma_count)
                if (
                    normalize_region >= 0
                    and region_graph.nodes[normalize_region][brain_id] > 0
                ):
                    somas_norm.append(
                        soma_count / region_graph.nodes[normalize_region][brain_id]
                    )
                else:
                    print(f"Warning: brain {brain_id} has no inputs from DRN")
                    somas_norm.append(0)
                somas_pct.append(
                    region_graph.nodes[region][brain_id]
                    / id_to_somatotals[brain_id]
                    * 100
                )
                subtypes.append(subtype + f" (n={subtype_counts[subtype]})")
                region_name.append(region_graph.nodes[region]["name"])
                brain_ids_data.append(brain_id)

        for region_component_name in composite_regions.keys():
            print(f"Populating: " + region_component_name)
            region_components = composite_regions[region_component_name]
            for brain_id in brain_ids:
                subtype = self.brain2paths[brain_id]["subtype"]
                soma_count = 0

                for region_component in region_components:
                    soma_count += region_graph.nodes[region_component][brain_id]

                somas.append(soma_count)

                if (
                    normalize_region >= 0
                    and region_graph.nodes[normalize_region][brain_id] > 0
                ):
                    somas_norm.append(
                        soma_count / region_graph.nodes[normalize_region][brain_id]
                    )
                else:
                    print(f"Warning: brain {brain_id} has no inputs from DRN")
                    somas_norm.append(0)

                somas_pct.append(soma_count / id_to_somatotals[brain_id] * 100)

                subtypes.append(subtype + f" (n={subtype_counts[subtype]})")
                region_name.append(region_component_name)
                brain_ids_data.append(brain_id)

            d = {
                "Somas (#)": somas,
                "Percent of Total Somas (%)": somas_pct,
                "Subtype": subtypes,
                "Region": region_name,
                "Brain ID": brain_ids_data,
            }
            if normalize_region >= 0:
                d["Normalized Somas"] = somas_norm

            df = pd.DataFrame(data=d)
            return df

    def _configure_annotator(self, df, axis, ind_variable: str):
        subtype_counts = self._get_subtype_counts()
        test = "Mann-Whitney"
        # my_logttest = StatTest(self._log_ttest_ind, test_long_name='Log t-test_ind', test_short_name='log-t')
        # test = "t-test_ind"
        correction = "fdr_by"

        pairs = []
        unq_subregions = df["Region"].unique()
        subtypes = df["Subtype"].unique()
        subtype_pairs = [
            (a, b)
            for idx, a in enumerate(subtypes)
            for b in subtypes[idx + 1 :]
            if subtype_counts[a.split("(")[0].strip()] > 1
            and subtype_counts[b.split("(")[0].strip()] > 1
        ]

        for subtype_pair in subtype_pairs:
            for subregion in unq_subregions:
                pairs.append(
                    (
                        (subregion, subtype_pair[0]),
                        (subregion, subtype_pair[1]),
                    )
                )

        fig_args = {
            "y": ind_variable,
            "x": "Region",
            "hue": "Subtype",
            "data": df,
        }

        annotator = Annotator(axis, pairs, **fig_args)
        annotator.configure(
            test=test,
            text_format="star",
            loc="outside",
            comparisons_correction=correction,
        )

        return annotator


def _log_ttest_ind(group_data1, group_data2, verbose=1, **stats_params):
    group_data1_log = np.log(group_data1)
    group_data2_log = np.log(group_data2)

    return ttest_ind(group_data1_log, group_data2_log, **stats_params)


def _get_corners_collection(
    vol_mask, vol_reg, block_size, max_coords: list = [-1, -1, -1]
):
    corners = _get_corners(vol_mask.shape, chunk_size=block_size, max_coords=max_coords)

    new_corners = []
    for corner in corners:
        x = corner[0][0]
        x2 = corner[1][0]
        x_reg = int(x / 8)
        x2_reg = np.amin([int(x2 / 8), vol_reg.shape[0]])
        y = corner[0][1]
        y2 = corner[1][1]
        y_reg = int(y / 8)
        y2_reg = np.amin([int(y2 / 8), vol_reg.shape[0]])
        z = corner[0][2]
        z2 = corner[1][2]

        new_corners.append(
            [[x_reg, y_reg, z], [x2_reg, y2_reg, z2], corner[0], corner[1]]
        )

    return new_corners


def _compute_composition_corner(corners, outdir, dir_base):
    l_c1 = corners[0]
    l_c2 = corners[1]
    m_c1 = corners[2]
    m_c2 = corners[3]

    fname = outdir + str(l_c1[0]) + "_" + str(l_c1[1]) + "_" + str(l_c1[2]) + ".pickle"
    if os.path.exists(fname):
        return

    dir = dir_base + "axon_mask"
    vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=False)

    dir = dir_base + "atlas_to_target"
    vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)

    try:
        labels = vol_reg[l_c1[0] : l_c2[0], l_c1[1] : l_c2[1], l_c1[2] : l_c2[2]]
        labels = np.repeat(np.repeat(labels, 8, axis=0), 8, axis=1)
        mask = vol_mask[m_c1[0] : m_c2[0], m_c1[1] : m_c2[1], m_c1[2] : m_c2[2]]
    except exceptions.EmptyVolumeException:
        return

    width = np.amin([mask.shape[0], labels.shape[0]])
    height = np.amin([mask.shape[1], labels.shape[1]])
    mask = mask[:width, :height, :]
    labels = labels[:width, :height, :]

    labels_unique = np.unique(labels[labels > 0])

    volumes = {}
    for unq in labels_unique:
        cur_vol = np.sum(mask[labels == unq])
        cur_total = np.sum(labels == unq)
        volumes[unq] = [cur_total, cur_vol]

    with open(fname, "wb") as f:
        pickle.dump(volumes, f)


def _combine_regional_segmentations(outdir):
    files = os.listdir(outdir)
    volumes = {}
    for file in tqdm(files, desc="Assembling results"):
        if "pickle" in file:
            filename = outdir + file
            with open(filename, "rb") as f:
                result = pickle.load(f)
            for key in result.keys():
                addition = result[key]
                if key in volumes.keys():
                    cur_vol = volumes[key][1]
                    cur_total = volumes[key][0]
                else:
                    cur_vol = 0
                    cur_total = 0

                cur_vol += addition[1]
                cur_total += addition[0]
                volumes[key] = [cur_total, cur_vol]
    return volumes


def collect_regional_segmentation(
    brain_id: str,
    data_file: str,
    outdir: str,
    ncpu: int = 1,
    max_coords: list = [-1, -1, -1],
):
    """Combine segmentation and registration to generate counts of axon voxels across brain regions. Note this scripts writes a file for every chunk, which might be many.

    Args:
        brain_id (str): Brain ID to process, from axon_data.py
        data_file (str): path to json file with data information.
        outdir (str): Path to directory to write files.
        ncpu (int, optional): Number of cpus to use for parallel processing. Defaults to 1.
        max_coords (list, optional): Upper limits of brain to complete processing, -1 will lead to processing the whole axis. Defaults to [-1, -1, -1].
    """
    with open(data_file) as f:
        data = json.load(f)
    brain2paths = data["brain2paths"]
    dir_base = brain2paths[brain_id]["base"]

    dir = os.path.join(dir_base, "axon_mask")
    vol_mask = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
    print(f"Mask shape: {vol_mask.shape}")

    dir = os.path.join(dir_base, "atlas_to_target")
    vol_reg = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
    print(f"Atlas shape: {vol_reg.shape}")

    corners = _get_corners_collection(
        vol_mask, vol_reg, block_size=[256, 256, 256], max_coords=max_coords
    )
    Parallel(n_jobs=ncpu)(
        delayed(_compute_composition_corner)(corner, outdir, dir_base)
        for corner in tqdm(corners, desc="Finding labels")
    )

    volumes = _combine_regional_segmentations(outdir)

    outpath = outdir + "wholebrain_" + brain_id + ".pkl"
    with open(outpath, "wb") as f:
        pickle.dump(volumes, f)


class AxonDistribution(BrainDistribution):
    """Generates visualizations of results of axon segmentations of a set of brains. Implements BrainDistribution.

    Arguments:
        brain_ids (list): List of brain IDs (keys of data json file).
        data_files (str): Path to json file that contains information about samples.
        regional_distribution_dir (str): Path to directory with pkl files that countain segmentation results by region.
        show_plots (bool): Whether to show plots, only works if you have graphics access.

    Attributes:
        regional_distribution_dir (str): Path to directory with pkl files that countain segmentation results by region.
        region_graph (nx.DiGraph): Graph of region hierarchy with segmentation results as node attributes.
        total_axon_vols (dict): Key - sample ID Value - Total volume of segmented axon.
        brain2paths (dict): Information about different data samples.
        show_plots (bool): Whether to show plots, only works if you have graphics access.

    """

    def __init__(
        self,
        brain_ids: list,
        data_file: str,
        regional_distribution_dir: str,
        show_plots: bool = True,
    ):
        super().__init__(brain_ids, data_file)
        self.regional_distribution_dir = regional_distribution_dir
        region_graph, total_axon_vols = self._setup_regiongraph(
            regional_distribution_dir
        )
        self.region_graph, self.total_axon_vols = region_graph, total_axon_vols
        self.show_plots = show_plots

    def _setup_regiongraph(self, regional_distribution_dir):
        regional_distribution_dir = self.regional_distribution_dir
        brain_ids = self.brain_ids
        region_graph = _setup_atlas_graph()
        max_level = 0

        # set to 0
        for node in region_graph.nodes:
            if region_graph.nodes[node]["level"] > max_level:
                max_level = region_graph.nodes[node]["level"]
            for brain_id in brain_ids:
                region_graph.nodes[node][brain_id + " axon"] = 0
                region_graph.nodes[node][brain_id + " total"] = 0

        # add data
        for brain_id in brain_ids:
            region_distribution = (
                regional_distribution_dir + "wholebrain_" + brain_id + ".pkl"
            )

            with open(region_distribution, "rb") as f:
                quantification_dict = pickle.load(f)

            for region in quantification_dict.keys():
                if region in region_graph.nodes:
                    region_graph.nodes[region][brain_id + " axon"] = region_graph.nodes[
                        region
                    ][brain_id + " axon"] + float(quantification_dict[region][1])
                    region_graph.nodes[region][
                        brain_id + " total"
                    ] = region_graph.nodes[region][brain_id + " total"] + float(
                        quantification_dict[region][0]
                    )

        total_axon_vols = {}

        for brain in brain_ids:
            total = 0
            for node in region_graph.nodes:
                total += region_graph.nodes[node][brain + " axon"]
            total_axon_vols[brain] = total

        # propagate counts up the hierarchy
        for brain_id in brain_ids:
            for lvl in range(max_level, 0, -1):
                for node in region_graph.nodes:
                    if region_graph.nodes[node]["level"] == lvl:
                        parent = list(region_graph.in_edges(node))[0][0]
                        region_graph.nodes[parent][brain_id + " axon"] = (
                            region_graph.nodes[parent][brain_id + " axon"]
                            + region_graph.nodes[node][brain_id + " axon"]
                        )
                        region_graph.nodes[parent][brain_id + " total"] = (
                            region_graph.nodes[parent][brain_id + " total"]
                            + region_graph.nodes[node][brain_id + " total"]
                        )

        return region_graph, total_axon_vols

    def napari_coronal_section(
        self, z: int, subtype_colors: dict, fold_on: bool = False
    ):
        """Generate napari viewer with allen parcellation and heat map of axon segmentations.

        Args:
            z (int): Index of coronal slice of allen atlas.
            subtype_colors (dict): Mapping of subtype to color to be used in visualization.
            fold_on (bool, optional): Whether to plot a single hemisphere, with results from other side mirrored. Defaults to False.
        """

        brain2paths = self.brain2paths
        if "filepath" in brain2paths["atlas"].keys():
            vol_atlas = io.imread(brain2paths["atlas"]["filepath"])
        else:
            vol_atlas = CloudVolume(brain2paths["atlas"]["url"])

        slice = np.squeeze(np.array(vol_atlas[z, :, :]))

        newslice, borders, half_width = self._slicetolabels(slice, fold_on=fold_on)

        if self.show_plots:
            v = napari.Viewer()
            v.add_labels(newslice, scale=[10, 10])

        heatmaps = {subtype: 0 * newslice for subtype in subtype_colors.keys()}
        for brain_id in self.brain_ids:
            print(brain_id)
            subtype = brain2paths[brain_id]["subtype"]

            transformed_mask_vol = CloudVolume(
                brain2paths[brain_id]["transformed_mask"], fill_missing=True
            )
            mask_slice = np.squeeze(transformed_mask_vol[z - 10 : z + 10, :, :])
            mask_slice = ndi.gaussian_filter(mask_slice.astype(float), sigma=(3, 3, 3))[
                10, :, :
            ]
            if fold_on:
                mask_slice = _fold(mask_slice)
            mask_slice[newslice == 0] = 0

            heatmaps[subtype] = heatmaps[subtype] + mask_slice

        for subtype in heatmaps.keys():
            heatmaps[subtype] = heatmaps[subtype] / np.amax(heatmaps[subtype])

        rgb_heatmap = [-1, -1, -1]
        for subtype in heatmaps.keys():
            if subtype_colors[subtype] == "red":
                rgb_heatmap[0] = heatmaps[subtype]
            elif subtype_colors[subtype] == "green":
                rgb_heatmap[1] = heatmaps[subtype]
            elif subtype_colors[subtype] == "blue":
                rgb_heatmap[2] = heatmaps[subtype]
        rgb_heatmap = [0 * newslice if type(i) == int else i for i in rgb_heatmap]

        rgb_heatmap = np.stack(rgb_heatmap, axis=-1)

        if self.show_plots:
            v.add_image(rgb_heatmap, rgb=True, scale=[10, 10], name=f"{subtype_colors}")
            v.add_labels(borders * 2, scale=[10, 10], name=f"z={z}")
            v.scale_bar.unit = "um"
            v.scale_bar.visible = True
            napari.run()

    def brainrender_axons(self, subtype_colors: dict, brain_region: str = "DR"):
        """Generate brainrender view to show axon segmentations.

        Args:
            subtype_colors (dict): Mapping of subtype to color to be used in visualization.
        """
        brain_ids = self.brain_ids
        brain2paths = self.brain2paths

        scene = Scene(atlas_name="allen_mouse_50um", title="Input Somas")
        scene.add_brain_region(brain_region, alpha=0.15)

        for subtype in subtype_colors.keys():
            im_total = None
            for i, brain_id in enumerate(brain_ids):
                if brain2paths[brain_id]["subtype"] == subtype:
                    print(f"Downloading transformed_mask from brain: {brain_id}")
                    vol = CloudVolume(
                        brain2paths[brain_id]["transformed_mask"], fill_missing=True
                    )
                    if im_total == None:
                        im_total = np.array(vol[:, :, :, :])
                    else:
                        im_total += np.array(vol[:, :, :, :])

            im_total = np.squeeze(im_total)
            im_total = np.swapaxes(im_total, 0, 2)

            # make a volume actor and add
            actor = Volume(
                im_total,
                voxel_size=20,  # size of a voxel's edge in microns
                as_surface=False,  # if true a surface mesh is rendered instead of a volume
                c=subtype_colors[
                    subtype
                ],  # use matplotlib colormaps to color the volume
            )

            scene.add(actor)

        # render
        scene.content

        if self.show_plots:
            scene.render()

    def region_barchart(
        self, regions: list, composite_regions: dict = {}, normalize_region: int = -1
    ):
        """Generate bar charts with statistical tests to compare segmentations between brains.

        Args:
            regions (list): List of Allen atlas brain region IDs to display data for (ID's found here: http://api.brain-map.org/api/v2/structure_graph_download/1.json)
            composite_regions (dict, optional): Mapping from a custom composite region (str, e.g. "Amygdala") to a set of regions that compose it (list of ints e.g. [131, 295, 319, 780]). Defaults to {}.
            normalize_region (int, optional): Region ID to normalize data for the normalized bar chart. Defaults to -1.
        """
        subtype_counts = self._get_subtype_counts()
        print(subtype_counts)

        df = self._make_bar_df(
            regions, composite_regions, subtype_counts, normalize_region
        )
        subtypes = df["Subtype"].unique()

        if normalize_region >= 0:
            fig, axes = plt.subplots(1, 3, figsize=(39, 20))
        else:
            fig, axes = plt.subplots(1, 2, figsize=(26, 20))
        sns.set(font_scale=2)

        # first panel
        fig_args = {
            "x": "Axon Density (%)",
            "y": "Region",
            "hue": "Subtype",
            "data": df,
        }

        sns.set(font_scale=2)
        bplot = sns.barplot(ax=axes[0], orient="h", **fig_args)
        bplot.set_xscale("log")

        if len(subtypes) > 1:
            annotator = self._configure_annotator(df, axes[0], "Axon Density (%)")
            annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
            annotator.apply_and_annotate()

        # second panel
        fig_args = {
            "x": "Percent Total Axon Volume (%)",
            "y": "Region",
            "hue": "Subtype",
            "data": df,
        }

        bplot = sns.barplot(ax=axes[1], orient="h", **fig_args)
        bplot.set_xscale("log")

        if len(subtypes) > 1:
            annotator = self._configure_annotator(
                df, axes[1], "Percent Total Axon Volume (%)"
            )
            annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
            annotator.apply_and_annotate()

        if normalize_region >= 0:
            # third panel
            fig_args = {
                "x": "Normalized Axon Density",
                "y": "Region",
                "hue": "Subtype",
                "data": df,
            }

            sns.set(font_scale=2)
            bplot = sns.barplot(ax=axes[2], orient="h", **fig_args)
            bplot.set_xscale("log")

            if len(subtypes) > 1:
                annotator = self._configure_annotator(
                    df, axes[2], "Normalized Axon Density"
                )
                annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
                annotator.apply_and_annotate()

        fig.tight_layout()

        if self.show_plots:
            plt.show()

    def _make_bar_df(
        self, regions, composite_regions, subtype_counts, normalize_region
    ):
        region_graph = self.region_graph
        brain_ids = self.brain_ids
        total_axon_vols = self.total_axon_vols

        subtypes = []
        axon_vols = []
        axon_denss_norm = []
        axon_denss = []
        region_name = []
        brain_ids_data = []

        for region in regions:
            print(f"Populating: {region_graph.nodes[region]['name']}")
            for brain_id in brain_ids:
                subtype = self.brain2paths[brain_id]["subtype"]
                axon_vol = region_graph.nodes[region][brain_id + " axon"]
                total_vol = region_graph.nodes[region][brain_id + " total"]

                if (
                    normalize_region >= 0
                    and region_graph.nodes[normalize_region][brain_id + " axon"] > 0
                ):
                    norm_factor = (
                        region_graph.nodes[normalize_region][brain_id + " axon"]
                        / region_graph.nodes[normalize_region][brain_id + " total"]
                    )
                else:
                    norm_factor = 1
                    if normalize_region >= 0:
                        print(
                            f"Warning: brain {brain_id} has no projection in normalizing region: {normalize_region}"
                        )

                if total_vol == 0 and axon_vol == 0:
                    axon_denss.append(0)
                    axon_denss_norm.append(0)
                elif total_vol == 0:
                    raise ValueError("positive axon volume in zero volume region?")
                else:
                    dens = axon_vol / total_vol
                    axon_denss.append(dens * 100)
                    axon_denss_norm.append(dens / norm_factor)

                axon_vols.append(axon_vol / total_axon_vols[brain_id] * 100)
                subtypes.append(subtype + f" (n={subtype_counts[subtype]})")
                region_name.append(region_graph.nodes[region]["name"])
                brain_ids_data.append(brain_id)

        for region_component_name in composite_regions.keys():
            print(f"Populating: " + region_component_name)
            region_components = composite_regions[region_component_name]
            for brain_id in brain_ids:
                subtype = self.brain2paths[brain_id]["subtype"]
                axon_vol = 0
                total_vol = 0

                for region_component in region_components:
                    axon_vol += region_graph.nodes[region_component][brain_id + " axon"]
                    total_vol += region_graph.nodes[region_component][
                        brain_id + " total"
                    ]

                if (
                    normalize_region >= 0
                    and region_graph.nodes[normalize_region][brain_id + " axon"] > 0
                ):
                    norm_factor = (
                        region_graph.nodes[normalize_region][brain_id + " axon"]
                        / region_graph.nodes[normalize_region][brain_id + " total"]
                    )
                else:
                    norm_factor = 1
                    if normalize_region >= 0:
                        print(
                            f"Warning: brain {brain_id} has no projection in normalizing region: {normalize_region}"
                        )

                if total_vol == 0 and axon_vol == 0:
                    axon_denss.append(0)
                    axon_denss_norm.append(0)
                elif total_vol == 0:
                    raise ValueError("positive axon volume in zero volume region?")
                else:
                    dens = axon_vol / total_vol
                    axon_denss.append(dens * 100)
                    axon_denss_norm.append(dens / norm_factor)

                axon_vols.append(axon_vol / total_axon_vols[brain_id] * 100)
                subtypes.append(subtype + f" (n={subtype_counts[subtype]})")
                region_name.append(region_component_name)
                brain_ids_data.append(brain_id)

            d = {
                "Percent Total Axon Volume (%)": axon_vols,
                "Axon Density (%)": axon_denss,
                "Subtype": subtypes,
                "Region": region_name,
                "Brain ID": brain_ids_data,
            }
            if normalize_region >= 0:
                d["Normalized Axon Density"] = axon_denss_norm

            df = pd.DataFrame(data=d)

            return df

    def _configure_annotator(self, df, axis, ind_variable: str):
        subtype_counts = self.subtype_counts

        test = StatTest(
            _log_ttest_ind, test_long_name="Log t-test_ind", test_short_name="log-t"
        )
        # test = "t-test_ind"
        correction = "fdr_by"

        pairs = []
        unq_subregions = df["Region"].unique()
        subtypes = df["Subtype"].unique()
        subtypes = [
            subtype
            for subtype in subtypes
            if subtype_counts[subtype.split(" (n=")[0]] > 1
        ]

        subtype_pairs = [
            (a, b) for idx, a in enumerate(subtypes) for b in subtypes[idx + 1 :]
        ]
        print(subtype_pairs)

        for subtype_pair in subtype_pairs:
            for subregion in unq_subregions:
                pairs.append(
                    (
                        (subregion, subtype_pair[0]),
                        (subregion, subtype_pair[1]),
                    )
                )

        fig_args = {
            "y": ind_variable,
            "x": "Region",
            "hue": "Subtype",
            "data": df,
        }

        annotator = Annotator(axis, pairs, **fig_args)
        annotator.configure(
            test=test,
            text_format="star",
            loc="outside",
            comparisons_correction=correction,
        )

        return annotator
