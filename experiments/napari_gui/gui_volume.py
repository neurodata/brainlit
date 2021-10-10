from mouselight_code.src import dynamic_programming_connected_double_learn, image_process
# import pandas as pd
from pathlib import Path
from skimage import io, measure
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
import numpy as np 
import napari
import pickle
import networkx as nx 
import scipy.ndimage as ndi 
from tqdm import tqdm
import random
# from scipy import stats
from mouselight_code.src.swc2voxel import Bresenham3D
import subprocess
import h5py
from napari_animation import AnimationWidget
import similaritymeasures
from cloudvolume import Skeleton
import re 

num = 0
path = '/Users/thomasathey/Documents/mimlab/mouselight/input/images/first10_quantitative/viterbi_'+str(num)+'.pickle'
path = '/Users/thomasathey/Documents/mimlab/mouselight/input/images/gui/viterbi_250.pickle'

with open(path, 'rb') as handle:
    viterbi = pickle.load(handle)

im = viterbi.image
new_labels = viterbi.labels


viewer = napari.Viewer(ndisplay=3)
viewer.add_image(im, name="image")
labels_layer = viewer.add_labels(new_labels, name="labels")
viewer.camera.angles = [0, -90, 180]

def get_layer_labels():
    labels = []
    label2state = {}
    label2layers = {}

    for i in range(len(viewer.layers)):
        if type(viewer.layers[i]) == napari.layers.points.points.Points:
            label_name = viewer.layers[i].name.split(" ")
            label = int(label_name[1])
            state = int(label_name[3])

            if label not in labels:
                labels.append(label)
                label2layers[label] = [i]
            else:
                label2layers[label] = label2layers[label] + [i]

            label2state[label] = state
    return labels, label2state, label2layers

@labels_layer.mouse_drag_callbacks.append
def select_state(layer, event):
    data_coordinates = layer.world_to_data(event.position)
    cords = np.round(data_coordinates).astype(int)
    val = layer.get_value(
        position=event.position,
        view_direction=event.view_direction,
        dims_displayed=event.dims_displayed,
        world=True)
    if val is None:
        return
    if val != 0: 
        existing_labels, label2state, label2layers = get_layer_labels()

        if val in existing_labels and label2state[val] == 0 and len(viterbi.comp_to_states[val]) > 1:
            state_num = 1
        else:
            state_num = 0

        state = viterbi.comp_to_states[val][state_num]
        if viterbi.state_to_comp[state][0] == "fragment":
            pt1 = viterbi.state_to_comp[state][2]["coord1"]
            pt2 = viterbi.state_to_comp[state][2]["coord2"]
            viewer.add_points([pt1], face_color='red', size=7, name=f"label {val} state {state_num} start")
        else:
            pt2 = viterbi.soma_locs[val][0,:]
        viewer.add_points([pt2], face_color='orange', size=7, name=f"label {val} state {state_num} end")

        msg = (
            f'clicked  on component {val} which is now is displaying endpoints: {pt2}'
        )
        print(msg)

        if val in existing_labels:
            layers = label2layers[val]
            layers.reverse()
            for layer in layers:
                viewer.layers.pop(layer)


def drawpath(state1, state2):
    path_states = nx.shortest_path(viterbi.nxGraph, state1, state2, weight='weight')

    path_comps = []
    for state in path_states:
        path_comps.append(viterbi.state_to_comp[state][1])
    print(f'path sequence: {path_states}')
    print(f'component sequence: {path_comps}')

    path_mask = 0*new_labels
    for i, label in enumerate(path_comps):
        path_mask[new_labels == label] = i+1

    lines = []
    cumul_cost = 0
    for s, state in enumerate(path_states):
        if s>0:
            dist_cost = viterbi.cost_mat_dist[path_states[s-1], state]
            int_cost = viterbi.cost_mat_int[path_states[s-1], state]
            cumul_cost += dist_cost + int_cost
            print(f"Trans. #{s}: dist cost state {path_states[s-1]}->state {state}, comp {viterbi.state_to_comp[path_states[s-1]][1]}->comp {viterbi.state_to_comp[state][1]}: {dist_cost:.2f}, int cost: {int_cost:.2f}, cum. cost: {cumul_cost:.2f}")
        if viterbi.state_to_comp[state][0] == "fragment":
            lines.append(list(viterbi.state_to_comp[state][2]["coord1"]))
            lines.append(list(viterbi.state_to_comp[state][2]["coord2"]))
        elif viterbi.state_to_comp[path_states[s-1]][0] == "fragment":
            lines.append(list(viterbi.state_to_comp[path_states[s-1]][2]["soma connection point"])) 

    return lines

@viewer.bind_key('t')
def accept_image(viewer):
    existing_labels, label2state, label2layers = get_layer_labels()
    if len(existing_labels) >= 2:
        state1 = viterbi.comp_to_states[existing_labels[-2]][label2state[existing_labels[-2]]]
        state2 = viterbi.comp_to_states[existing_labels[-1]][label2state[existing_labels[-1]]]

        lines = drawpath(state1, state2)
        viewer.add_shapes(lines, shape_type="path", edge_color="blue", edge_width=1)

        layers  = label2layers[existing_labels[-2]] + label2layers[existing_labels[-1]]
        layers.sort(reverse=True)

        for layer in layers:
            viewer.layers.pop(layer)
    else:
        print("Not enough states selected")

@viewer.bind_key('c')
def accept_image(viewer):
    while len(viewer.layers) > 2:
        viewer.layers.pop(-1)
    
napari.run()