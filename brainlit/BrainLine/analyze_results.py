from brainlit.BrainLine.data.soma_data import brain2paths, brain2centers
import numpy as np
from cloudreg.scripts.transform_points import NGLink
from cloudvolume import CloudVolume
from tqdm import tqdm
from skimage import io, measure
from brainlit.BrainLine.util import (
    json_to_points,
    find_atlas_level_label,
    fold,
    setup_atlas_graph,
    get_atlas_level_nodes,
)
import napari
import scipy.ndimage as ndi
import seaborn as sns
from statannotations.Annotator import Annotator
import pandas as pd
import matplotlib.pyplot as plt

class SomaDistribution:
    def __init__(self, brain_ids: list):
        self.brain_ids = brain_ids
        atlas_points = self._retrieve_soma_coords(brain_ids)
        self.atlas_points = atlas_points
        id_to_regioncounts = self._get_regions(atlas_points)
        self.id_to_regioncounts = id_to_regioncounts
        region_graph = self._setup_regiongraph()
        self.region_graph = region_graph



    def _retrieve_soma_coords(self, brain_ids: list):
        atlas_points = {}
        for brain_id in brain_ids:
            if "somas_atlas_url" in brain2paths[brain_id].keys():
                viz_link = brain2paths[brain_id]["somas_atlas_url"]
                viz_link = NGLink(viz_link.split("json_url=")[-1])
                ngl_json = viz_link._json
                for layer in ngl_json["layers"]:
                    if layer["type"] == "annotation":
                        points = []
                        for annot in layer["annotations"]:
                            points.append(annot["point"])

                        atlas_points[brain_id] = np.array(points)
                        print(
                            f'Brain {brain_id}: Collecting atlas space soma points from layer: {layer["name"]}'
                        )
                        break
            else:
                print(f"No somas_atlas_url layer for brain: {brain_id}")
            
        return atlas_points
    
    def _get_regions(self, points: dict):
        if "filepath" in brain2paths["atlas"].keys():
            vol_atlas = io.imread(brain2paths["atlas"]["filepath"])
        else:
            vol_atlas = CloudVolume(brain2paths["atlas"]["url"])

        id_to_regioncounts = {}
        for brain_id in points.keys():
            brain_dict = {}
            for point in points[brain_id]:
                point_int = [int(np.round(p)) for p in point]
                region = vol_atlas[point_int[0], point_int[1], point_int[2]]
                if region in brain_dict.keys():
                    brain_dict[region] = brain_dict[region] + 1
                else:
                    brain_dict[region] = 1

            id_to_regioncounts[brain_id] = brain_dict

        return id_to_regioncounts


    def napari_coronal_section(self, z: int, subtype_colors: dict, symbols: list, fold_on: bool = False):
        atlas_points = self.atlas_points
        if "filepath" in brain2paths["atlas"].keys():
            vol_atlas = io.imread(brain2paths["atlas"]["filepath"])
        else:
            vol_atlas = CloudVolume(brain2paths["atlas"]["url"])

        slice = vol_atlas[z, :, :]

        newslice, borders, half_width = self._slicetolabels(slice, fold_on = fold_on)

        v = napari.Viewer()
        v.add_labels(newslice, scale=[10, 10])
        v.add_image(borders, scale=[10, 10], name=f"z={z}")

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

            v.add_points(
                fg_points,
                symbol=symbols[gtype_counts[gtype]],
                face_color=subtype_colors[gtype],
                size=10,
                name=f"{key}: {gtype}",
                scale=[10, 10],
            )
            gtype_counts[gtype] = gtype_counts[gtype] + 1

        v.scale_bar.unit = "um"
        v.scale_bar.visible = True
        napari.run()


    def _slicetolabels(self, slice, fold_on: bool = False, atlas_level: int = 5):
        region_graph = setup_atlas_graph()
        atlas_level_nodes = get_atlas_level_nodes(atlas_level, region_graph)
        newslice = np.copy(slice)
        new_labels = {}

        for label in tqdm(np.unique(slice), desc=f"Relabeling slice"):
            atlas_level_label = find_atlas_level_label(
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

    def region_barchart(self, regions: list, composite_regions: dict = {}, normalize_region: int = -1):
        region_graph = self.region_graph

        subtype_counts = self._get_subtype_counts()
        id_to_somatotals = self._count_somas()

        df = self._make_bar_df(regions, composite_regions, id_to_somatotals, subtype_counts, normalize_region)
        subtypes = df["Subtype"].unique()

        fig, axes = plt.subplots(1, 3, figsize=(39, 13))
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
            "x": "Normalized Somas",
            "y": "Region",
            "hue": "Subtype",
            "data": df,
        }

        sns.set(font_scale=2)
        bplot = sns.barplot(ax=axes[1], orient="h", **fig_args)
        bplot.set_xscale("log")
        
        if len(subtypes) > 1:
            annotator = self._configure_annotator(df, axes[1], "Normalized Somas")
            annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
            annotator.apply_and_annotate()

        # third panel
        fig_args = {
            "x": "Percent of Total Somas (%)",
            "y": "Region",
            "hue": "Subtype",
            "data": df,
        }

        bplot = sns.barplot(ax=axes[2], orient="h", **fig_args)
        bplot.set_xscale("log")

        if len(subtypes) > 1:
            annotator = self._configure_annotator(df, axes[2], "Percent of Total Somas (%)")
            annotator.new_plot(bplot, orient="h", plot="barplot", **fig_args)
            annotator.apply_and_annotate()

        fig.tight_layout()
        plt.show()


    def _setup_regiongraph(self):
        brain_ids = self.brain_ids
        id_to_regioncounts = self.id_to_regioncounts
        region_graph = setup_atlas_graph()
        max_level = 0

        # set to 0
        for node in region_graph.nodes:
            if region_graph.nodes[node]["level"] > max_level:
                max_level = region_graph.nodes[node]["level"]
            for brain_id in brain_ids:
                region_graph.nodes[node][brain_id] = 0
        
        #add counts
        for brain_id in brain_ids:
            regioncounts = id_to_regioncounts[brain_id]
            for region in regioncounts.keys():
                if region in region_graph.nodes:
                    region_graph.nodes[region][brain_id] = region_graph.nodes[region][brain_id] + id_to_regioncounts[brain_id][region]

        #propagate counts up the hierarchy
        for brain_id in brain_ids:
            for lvl in range(max_level, 0, -1):
                for node in region_graph.nodes:
                    if region_graph.nodes[node]["level"] == lvl:
                        parent = list(region_graph.in_edges(node))[0][0]
                        region_graph.nodes[parent][brain_id] = region_graph.nodes[parent][brain_id] + region_graph.nodes[node][brain_id]

        return region_graph


    def _get_subtype_counts(self):
        brain_ids = self.brain_ids
        counts = {}
        for brain_id in brain_ids:
            subtype = brain2paths[brain_id]["subtype"]
            if subtype in counts.keys():
                counts[subtype] = counts[subtype] + 1
            else:
                counts[subtype] = 1
        return counts

    def _count_somas(self):
        id_to_somatotals = {}
        brain_ids = self.brain_ids
        atlas_points = self.atlas_points

        for brain_id in brain_ids:
            points = atlas_points[brain_id]
            id_to_somatotals[brain_id] = points.shape[0]
        return id_to_somatotals

    def _make_bar_df(self, regions: list, composite_regions, id_to_somatotals: dict, subtype_counts: dict, normalize_region):
        region_graph = self.region_graph
        brain_ids = self.brain_ids

        subtypes = []
        somas = []
        somas_norm = []
        somas_pct = []
        region_name = []
        brain_ids_data = []

        for region in regions:
            print(f"Populating: {region_graph.nodes[region]['name']}" )
            for brain_id in brain_ids:
                subtype = brain2paths[brain_id]["subtype"]
                soma_count = region_graph.nodes[region][brain_id]
                somas.append(soma_count)
                if normalize_region >= 0 and region_graph.nodes[normalize_region][brain_id] > 0:
                    somas_norm.append(soma_count / region_graph.nodes[normalize_region][brain_id])
                else:
                    print(f"Warning: brain {brain_id} has no inputs from DRN")
                    somas_norm.append(0)
                somas_pct.append(region_graph.nodes[region][brain_id] / id_to_somatotals[brain_id] * 100)
                subtypes.append(subtype + f" (n={subtype_counts[subtype]})")
                region_name.append(region_graph.nodes[region]["name"])
                brain_ids_data.append(brain_id)

        for region_component_name in composite_regions.keys():
            print(f"Populating: " + region_component_name)
            region_components = composite_regions[region_component_name]
            for brain_id in brain_ids:
                subtype = brain2paths[brain_id]["subtype"]
                soma_count = 0

                for region_component in region_components:
                    soma_count += region_graph.nodes[region_component][brain_id]

                somas.append(soma_count)

                if normalize_region >= 0 and region_graph.nodes[normalize_region][brain_id] > 0:
                    somas_norm.append(soma_count / region_graph.nodes[normalize_region][brain_id])
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
        test = "Mann-Whitney"
        # test = "t-test_ind"
        correction = "fdr_by"

        pairs = []
        unq_subregions = df["Region"].unique()
        subtypes = df["Subtype"].unique()
        subtype_pairs = [(a, b) for idx, a in enumerate(subtypes) for b in subtypes[idx + 1 :]]

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
            test=test, text_format="star", loc="outside", comparisons_correction=correction
        )

        return annotator
    
class AxonDistribution:
    def __init__(self, brain_ids: list):
        self.brain_ids = brain_ids
