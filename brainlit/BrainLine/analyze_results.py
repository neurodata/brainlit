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

class SomaDistribution:
    def __init__(self, brain_ids: list):
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

        self.atlas_points = atlas_points

    def napari_coronal_section(self, z: int, subtype_colors: dict, symbols: list, fold_on: bool = False):
        atlas_points = self.atlas_points
        if "filepath" in brain2paths["atlas"].keys():
            vol_atlas = io.imread(brain2paths["atlas"]["filepath"])
        else:
            vol_atlas = CloudVolume(brain2paths["atlas"]["url"])

        slice = vol_atlas[z, :, :]

        newslice, borders, half_width = self.slicetolabels(slice, fold_on = fold_on)

        v = napari.Viewer()
        v.add_labels(newslice, scale=[10, 10])
        v.add_image(borders, scale=[10, 10], name=f"z={z}")

        gtype_counts = {}
        for key in subtype_colors.keys():
            gtype_counts[key] = 0

        for key in tqdm(atlas_points.keys(), desc=f"Processing brains in z={z}"):
            ra = atlas_points[key]
            gtype = brain2paths[key]["genotype"]

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
        


    def slicetolabels(self, slice, fold_on: bool = False, atlas_level: int = 5):
        G = setup_atlas_graph()
        atlas_level_nodes = get_atlas_level_nodes(atlas_level, G)
        newslice = np.copy(slice)
        new_labels = {}

        for label in tqdm(np.unique(slice), desc=f"Relabeling slice"):
            atlas_level_label = find_atlas_level_label(
                label, atlas_level_nodes, atlas_level, G
            )
            newslice[slice == label] = atlas_level_label
            if atlas_level_label not in new_labels.keys():
                if atlas_level_label in G.nodes:
                    name = G.nodes[atlas_level_label]["name"]
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

