import pickle
from sys import intern

from numpy import uint32
import numpy as np
import zarr
from napari_plugin_engine import napari_hook_implementation
from qtpy.QtWidgets import QWidget, QHBoxLayout, QPushButton
from magicgui import magic_factory
import pathlib
import napari


def viterbrain_reader(path: str) -> list:
    with open(path, "rb") as handle:
        viterbi = pickle.load(handle)

    layer_labels = zarr.open(viterbi.fragment_path)
    image_path = viterbi.fragment_path[:-12] + ".zarr"
    layer_image = zarr.open(image_path)

    scale = viterbi.resolution

    meta_labels = {"name": "fragments", "scale": scale}
    meta_image = {"name": "image", "scale": scale}

    return [(layer_image, meta_image, "image"), (layer_labels, meta_labels, "labels")]


def napari_get_reader(path: str) -> list:
    parts = path.split(".")
    if parts[-1] == "pickle" or parts[-1] == "pkl":
        return viterbrain_reader
    else:
        return None


@magic_factory(
    call_button="Trace", start_comp={"max": 2**20}, end_comp={"max": 2**20}
)
def comp_trace(
    v: napari.Viewer,
    start_comp: int,
    end_comp: int,
    filename=pathlib.Path("/some/path.pickle"),
) -> None:
    with open(filename, "rb") as handle:
        viterbi = pickle.load(handle)

    def comp2point(comp: int) -> list:
        state = viterbi.comp_to_states[comp][0]
        if viterbi.nxGraph.nodes[state]["type"] == "fragment":
            return viterbi.nxGraph.nodes[state]["point1"]
        else:
            coords = viterbi.soma_fragment2coords[comp]
            centroid = np.mean(coords, axis=0)
            centroid = [int(c) for c in centroid]
            return centroid

    start_pt = comp2point(start_comp)
    end_pt = comp2point(end_comp)

    print(f"tracing from {start_pt} to {end_pt}")

    path = viterbi.shortest_path(start_pt, end_pt)

    v.add_shapes(
        path,
        shape_type="path",
        edge_color="r",
        edge_width=1,
        name=f"trace {start_comp} to {end_comp}",
        scale=viterbi.resolution,
    )
