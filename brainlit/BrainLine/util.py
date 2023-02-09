import urllib
import json
import numpy as np
import os
import networkx as nx
from pathlib import Path
from brainlit.BrainLine.parse_ara import build_tree


def json_to_points(url, round=False):
    """Extract points from a neuroglancer url.

    Args:
        url (str): url to neuroglancer state (that was posted to neurodata json server).
        round (bool, optional): whether to round coordinates to integers. Defaults to False.

    Returns:
        dict: Keys are names of point layers and values are lists of points from that layer.
    """
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
    """Find file paths of samples in a given directory according to filters used in the workflow.


    Args:
        dir (str): path to directory.
        dset (str, optional): dataset type identifier. Defaults to "val".
        add_dir (bool, optional): whether output paths should include the directory path. Defaults to False.

    Returns:
        list: list of file path strings.
    """
    dir = Path(dir)
    items = os.listdir(dir)

    items = [item for item in items if ".h5" in item]
    items = [item for item in items if "Probabilities" not in item]
    items = [item for item in items if "Labels" not in item]
    items = [item for item in items if dset in item]

    if add_dir:
        items = [str(dir / item) for item in items]

    return items


def setup_atlas_graph():
    """Create networkx graph of regions in allen atlas (from ara_structure_ontology.json). Initially uses vikram's code in build_tree, then converts to networkx.

    Returns:
        nx.DiGraph: graph representing hierarchy of allen parcellation.
    """
    cd = Path(os.path.dirname(__file__))

    # create vikram object
    f = json.load(
        open(
            cd / "data" / "ara_structure_ontology.json",
            "r",
        )
    )

    tree = build_tree(f)
    stack = [tree]

    # create nx graph
    queue = [tree]
    cur_level = -1
    counter = 0
    G = nx.DiGraph()
    max_level = 0

    while len(queue) > 0:
        node = queue.pop(0)
        if node.level > max_level:
            max_level = node.level
        G.add_node(
            node.id,
            level=node.level,
            st_level=node.st_level,
            name=node.name,
            acronym=node.acronym,
            label=str(node.st_level) + ") " + node.name,
        )
        if node.parent_id is not None:
            G.add_edge(node.parent_id, node.id)

        queue += node.children

    return G


def get_atlas_level_nodes(atlas_level, atlas_graph):
    """Find regions in atlas that are at a specified level in the hierarchy

    Args:
        atlas_level (int): desired level in the atlas.
        atlas_graph (nx.DiGraph): graph of allen atlas, created from setup_atlas_graph.

    Returns:
        list: list of region ids at the desired hierarchy level.
    """
    atlas_level_nodes = []

    for node in atlas_graph.nodes:
        if atlas_graph.nodes[node]["st_level"] == atlas_level:
            atlas_level_nodes.append(node)
    return atlas_level_nodes


def find_atlas_level_label(label, atlas_level_nodes, atlas_level, G):
    """Map a given region label to a label at a specified level in the hierarchy.

    Args:
        label (int): region label.
        atlas_level_nodes (list): list of region IDs to which label will be mapped.
        atlas_level (int): level at which the atlas_level_nodes come from.
        G (nx.DiGraph): network of region hierarchy.

    Raises:
        ValueError: Found a node that has more than one parent (which is not possible for a tree).
        ValueError: Was not able to find a node at the desired atlas_level to map the label to.

    Returns:
        int: the relevant atlas label at the desired level in the hierarchy.
    """
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
    """Take a 2D image and add the left half to a reflected version of the right half.

    Args:
        image (nd.array): Image to be folded

    Returns:
        nd.array: Folded image.
    """
    half_width = np.round(image.shape[1] / 2).astype(int)
    left = image[:, :half_width]
    right = image[:, half_width:]
    left = left + np.flip(right, axis=1)
    return left
