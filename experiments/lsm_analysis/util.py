import urllib
import json
import numpy as np
import os
import networkx as nx


def json_to_points(url, round=False):
    pattern = "json_url="
    idx = url.find(pattern) + len(pattern)

    json_url = url[idx:]

    data = urllib.request.urlopen(json_url)

    string = data.readlines()[0].decode("utf-8")

    js = json.loads(string)

    point_layers = {}

    for layer in js["layers"]:
        if layer["type"] == "annotation":
            points = []
            for point in layer["annotations"]:
                coord = point["point"]
                if round:
                    coord = [int(np.round(c)) for c in coord]
                points.append(coord)
            point_layers[layer["name"]] = points
    return point_layers


def find_sample_names(dir, dset="val", add_dir=False):
    items = os.listdir(dir)

    items = [item for item in items if ".h5" in item]
    items = [item for item in items if "Probabilities" not in item]
    items = [item for item in items if "Labels" not in item]
    items = [item for item in items if dset in item]

    if add_dir:
        items = [dir + item for item in items]

    return items


def find_atlas_level_label(label, atlas_level_nodes, atlas_level, G):
    if label == 0 or label not in G.nodes or G.nodes[label]["st_level"] <= atlas_level:
        return label
    else:
        counter = 0
        # find which region of atlas_level is parent
        for atlas_level_node in atlas_level_nodes:
            if label in nx.algorithms.dag.descendants(G, source=atlas_level_node):
                counter += 1
                atlas_level_label = atlas_level_node
        if counter == 0:
            preds = list(G.predecessors(label))
            if len(preds) != 1:
                raise ValueError(f"{len(preds)} predecessors of node {label}")
            atlas_level_label = find_atlas_level_label(
                preds[0], atlas_level_nodes, atlas_level, G
            )
            counter += 1
        if counter != 1:
            raise ValueError(f"{counter} atlas level predecessors of {label}")
        return atlas_level_label


def fold(image):
    half_width = np.round(image.shape[1] / 2).astype(int)
    left = image[:, :half_width]
    right = image[:, half_width:]
    left = left + np.flip(right, axis=1)
    return left
