from pathlib import Path
from matplotlib.pyplot import colorbar
from networkx.algorithms.operators.unary import reverse
import numpy as np
import napari
import pickle
import networkx as nx
from scipy.fft import next_fast_len
from sklearn.manifold import trustworthiness
from tqdm import tqdm
import os
from napari_animation import AnimationWidget
from brainlit.utils.Neuron_trace import NeuronTrace
import zarr
from magicgui import widgets

scale = [0.3, 0.3, 1]
num = 0

root_dir = Path(os.path.abspath(""))
data_dir = os.path.join(root_dir, "data", "example")
path = os.path.join(data_dir, "3-1-soma_viterbrain.pickle")

with open(path, "rb") as handle:
    viterbi = pickle.load(handle)

im_path = os.path.join(data_dir, "3-1-soma.zarr")
z = zarr.open(im_path, "r")
im = z[:, :, :]
print(f"Image shape: {im.shape}")
fragment_path = os.path.join(data_dir, "3-1-soma_labels.zarr")
z = zarr.open(fragment_path, "r")
new_labels = z[:, :, :]
print(f"Labels shape: {new_labels.shape}")

state2centroid = {}
for soma_frag in viterbi.soma_fragment2coords.keys():
    state2centroid[soma_frag] = np.mean(viterbi.soma_fragment2coords[soma_frag], axis=0)

viewer = napari.Viewer(ndisplay=3)
viewer_copy = viewer  # added viewer copy to bypass boolean behavior from viewer after QPushButton emits signal
viewer.add_image(im, name="image", scale=scale)
# viewer.add_shapes(SNT, shape_type="path", edge_color="blue", edge_width=1)
labels_layer = viewer.add_labels(new_labels, name="labels", scale=scale)
animation_widget = AnimationWidget(viewer)
viewer.window.add_dock_widget(animation_widget, area="right")
trace = widgets.PushButton(text="trace")
switch = widgets.PushButton(text="switch states")
save = widgets.PushButton(text="save")
clear = widgets.PushButton(text="clear selected states")
clear_all = widgets.PushButton(text="clear all")
next = widgets.PushButton(text="next color")
container = widgets.Container(widgets=[trace, switch, save, clear, clear_all, next])
viewer.window.add_dock_widget(container, area="right")
# viewer.scale_bar.visible = True

colors = ["green", "blue", "red"]

keys = {
    "o": "switch state",
    "t": "trace",
    "s": "save",
    "c": "clear selected states",
    "q": "clear all annotations",
    "n": "next color",
}


def get_layers():
    state_layers = {}
    trace_layers = {}
    state_order = []
    soma_end = False

    for l in range(len(viewer.layers)):
        layer = viewer.layers[l]
        if (
            type(layer) == napari.layers.shapes.shapes.Shapes
            or type(layer) == napari.layers.points.points.Points
        ):
            label_name = layer.name.split(" ")
            if label_name[0] == "state":
                state = int(label_name[1])
                if state not in state_order:
                    state_order.append(state)

                if state not in state_layers.keys():
                    state_layers[state] = [l]
                else:
                    layer_list = [l] + state_layers[state]
                    state_layers[state] = layer_list
                if type(layer) == napari.layers.points.points.Points:
                    soma_end = True
            elif label_name[0] == "trace":
                trace_layers[l] = layer.data[0]

    return state_layers, trace_layers, state_order, soma_end


def remove_layers(layers):
    print(f"Removing layers: {layers}")
    layers.sort(reverse=True)
    for layer in layers:
        viewer.layers.pop(layer)


def draw_arrow(val, state):
    factor = 7
    if viterbi.nxGraph.nodes[state]["type"] == "fragment":
        pt1 = viterbi.nxGraph.nodes[state]["point1"]
        pt2 = viterbi.nxGraph.nodes[state]["point2"]

        orient = np.subtract(pt1, pt2) / np.linalg.norm(np.subtract(pt1, pt2))
        base = np.add(pt2, orient * factor)
        orient2 = orient.copy()
        for i, term in enumerate(orient):
            if term != 0:
                orient2[i] = -orient[i]
                break
        c1 = np.cross(orient, orient2) / np.linalg.norm(np.cross(orient, orient2))
        c2 = np.cross(orient, c1) / np.linalg.norm(np.cross(orient, c1))
        corner1 = np.add(base, factor * c1)
        corner2 = np.add(base, -factor * c1)
        poly1 = np.squeeze(np.stack([pt2, corner1, corner2], axis=0))
        corner1 = np.add(base, factor * c2)
        corner2 = np.add(base, -factor * c2)
        poly2 = np.squeeze(np.stack([pt2, corner1, corner2], axis=0))
        viewer.add_shapes(
            [pt1, pt2],
            shape_type="path",
            edge_color=colors[-1],
            edge_width=1,
            name=f"state {state} label {val} stem",
            scale=scale,
        )
        viewer.add_shapes(
            [poly1, poly2],
            shape_type="polygon",
            edge_color=colors[-1],
            face_color=colors[-1],
            edge_width=1,
            name=f"state {state} label {val} head",
            scale=scale,
        )
    else:
        pt2 = viterbi.soma_fragment2coords[val][0, :]
        viewer.add_points(
            [pt2],
            face_color=colors[-1],
            size=7,
            name=f"state {state} label {val} end",
            scale=scale,
        )
    return pt2


@viewer.mouse_drag_callbacks.append
def select_state(viewer, event):
    data_coordinates = labels_layer.world_to_data(event.position)
    cords = np.round(data_coordinates).astype(int)
    val = labels_layer.get_value(
        position=event.position,
        view_direction=event.view_direction,
        dims_displayed=event.dims_displayed,
        world=True,
    )
    if val is None:
        return
    elif val != 0:
        state = viterbi.comp_to_states[val][0]
        states, _, _, _ = get_layers()
        if state in states.keys():
            print(f"State {state} already selected")
        else:
            pt2 = draw_arrow(val, state)
            print(
                f"clicked  on component {val} which is now is displaying endpoints: {pt2}"
            )


@viewer.bind_key("o")
@switch.clicked.connect
def switch_state(viewer):
    states, _, state_order, _ = get_layers()
    last_state = state_order[-1]

    if last_state is not None:
        label = viterbi.nxGraph.nodes[last_state]["fragment"]
        if last_state == viterbi.comp_to_states[label][0]:
            new_state = viterbi.comp_to_states[label][1]
        else:
            new_state = viterbi.comp_to_states[label][0]

        pt2 = draw_arrow(label, new_state)

        print(f"switched component {label} which is now is displaying endpoints: {pt2}")

        layers = states[last_state]
        remove_layers(layers)
    else:
        print("No label selected")


def drawpath(state1, state2):
    path_states = nx.shortest_path(viterbi.nxGraph, state1, state2, weight="weight")

    path_comps = []
    for state in path_states:
        path_comps.append(viterbi.nxGraph.nodes[state]["fragment"])
    print(f"path sequence: {path_states}")
    print(f"component sequence: {path_comps}")

    path_mask = 0 * new_labels
    for i, label in enumerate(path_comps):
        path_mask[new_labels == label] = i + 1

    lines = []
    cumul_cost = 0
    for s, state in enumerate(path_states):
        if s > 0:
            dist_cost = viterbi.nxGraph.edges[path_states[s - 1], state]["dist_cost"]
            int_cost = viterbi.nxGraph.edges[path_states[s - 1], state]["int_cost"]
            cumul_cost += dist_cost + int_cost
            comp1 = viterbi.nxGraph.nodes[path_states[s - 1]]["fragment"]
            comp2 = viterbi.nxGraph.nodes[state]["fragment"]
            print(
                f"Trans. #{s}: dist cost state {path_states[s-1]}->state {state}, comp {comp1}->comp {comp2}: {dist_cost:.2f}, int cost: {int_cost:.2f}, cum. cost: {cumul_cost:.2f}"
            )
        if viterbi.nxGraph.nodes[state]["type"] == "fragment":
            lines.append(list(viterbi.nxGraph.nodes[state]["point1"]))
            lines.append(list(viterbi.nxGraph.nodes[state]["point2"]))
        elif viterbi.nxGraph.nodes[path_states[s - 1]]["type"] == "fragment":
            lines.append(list(viterbi.nxGraph.nodes[path_states[s - 1]]["soma_pt"]))
            soma_frag = viterbi.nxGraph.nodes[state]["fragment"]
            lines.append(list(state2centroid[soma_frag]))

    return lines


@viewer.bind_key("t")
@trace.clicked.connect
def trace(viewer):
    states, traces, state_order, soma_end = get_layers()
    if len(state_order) >= 2:
        state1 = state_order[-2]
        state2 = state_order[-1]
        print(f"Tracing from {state1} to {state2}")
        lines = drawpath(state1, state2)
        viewer_copy.add_shapes(
            lines,
            shape_type="path",
            edge_color=colors[-1],
            edge_width=1,
            name=f"trace {state1} to {state2}",
            scale=scale,
        )
        layers = states[state1]  # + states[state2]
        if soma_end:
            layers += states[state2]

        remove_layers(layers)
    else:
        print("Not enough states selected")


@viewer.bind_key("s")
@save.clicked.connect
def save_traces(viewer):
    _, traces, _, truth = get_layers()

    if len(traces.keys()) > 0:
        file_name = "/Users/sejalsrivastava/Desktop/traces.swc"
        with open(file_name, "w") as f:
            f.write("#idx \t x \t y \t z \t parent\n")
            idx = 0
            for trace in tqdm(traces.keys(), desc="Saving traces"):
                data = traces[trace]
                for row_num, row in enumerate(data):
                    if row_num == 0:
                        parent = -1
                    else:
                        parent = idx - 1
                    line = f"{idx} \t {row[0]} \t {row[1]} \t {row[2]} \t {parent}\n"
                    f.write(line)
                    idx += 1


@viewer.bind_key("c")
@clear.clicked.connect
def clear(viewer):
    layers_to_remove = []
    states, traces, _, _ = get_layers()
    for state in states.keys():
        layers_to_remove += states[state]
    remove_layers(layers_to_remove)


@viewer.bind_key("q")
@clear_all.clicked.connect
def clear_all(viewer):
    layers_to_remove = []
    states, traces, _, _ = get_layers()
    for state in states.keys():
        layers_to_remove += states[state]
    layers_to_remove += list(traces.keys())
    remove_layers(layers_to_remove)


@viewer.bind_key("n")
@next.clicked.connect
def clear_all(viewer):
    color = colors.pop()
    colors.insert(0, color)


napari.run()
