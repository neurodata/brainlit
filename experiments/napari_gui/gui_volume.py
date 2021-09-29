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
janelia = False

# read image
im_path = "/Users/thomasathey/Documents/mimlab/mouselight/input/images/first10_quantitative/images/2018-08-01_" + str(num) + "_first10_quantitative.tif"

im = Path(im_path)
im_og = io.imread(im, plugin="tifffile")
print(f"Image shape: {im_og.shape}")

# read coords
if janelia:
    csv_path = "/Users/thomasathey/Documents/mimlab/mouselight/input/images/first10_quantitative/voxel_coords.csv"
    coords = np.genfromtxt(csv_path, delimiter=',')
    coords = coords[10*num:10*(num+1)].astype(int)
    soma_coords = [list(coords[0,:])]
    axon_coords = [list(coords[-1,:])]
else:
    csv_path = "/Users/thomasathey/Documents/mimlab/mouselight/input/images/first10_quantitative/my_points/points_" + str(num) + ".csv"
    coords = np.genfromtxt(csv_path, delimiter=',')
    coords = coords[1:].astype(int)
    coords = coords[:,1:]
    soma_coords = [list(coords[-1,:])]
    axon_coords = [list(coords[0,:])]

coords_list = list(coords)
coords_list = [list(c) for c in coords_list]
print(f"coords shape: {coords.shape}")

# read ilastik
pred_path = "/Users/thomasathey/Documents/mimlab/mouselight/input/images/first10_quantitative/2018-08-01_" + str(num) + "_first10_quantitative_Probabilities.h5"
f = h5py.File(pred_path, 'r')
pred = f.get('exported_data')
pred = pred[:,:,:,1]
im_processed = pred



with open('/Users/thomasathey/Documents/mimlab/mouselight/input/images/first10_quantitative/viterbi_'+str(num)+'.pickle', 'rb') as handle:
    viterbi = pickle.load(handle)

im = viterbi.image_raw
im_og = im
im_processed = viterbi.image
new_labels = viterbi.labels
soma_lbls = viterbi.soma_lbls

_, axon_lbls = image_process.label_points(new_labels, axon_coords)
_, soma_lbls = image_process.label_points(new_labels, soma_coords)
soma_lbl = soma_lbls[0]
print(f"Axon labels: {axon_lbls}, soma labels: {soma_lbls}")

viewer = napari.Viewer(ndisplay=3)
viewer.add_image(viterbi.image_raw, name="image")
labels_layer = viewer.add_labels(new_labels, name="labels")
viewer.camera.angles = [0, -90, 180]

@labels_layer.mouse_drag_callbacks.append
def get_connected_component_shape(layer, event):
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
        print(viewer.layers)
        print(len(viewer.layers))
        while len(viewer.layers) > 2:
            viewer.layers.pop(-1)
        print(viewer.layers)
        print(len(viewer.layers))

        
        
        start1 = viterbi.comp_to_states[val][0]
        pt1 = viterbi.state_to_comp[start1][2]["coord1"]
        pt2 = viterbi.state_to_comp[start1][2]["coord2"]
        print(f"pt1: {pt1}")
        print(f"pt2: {pt2}")
        viewer.add_image(viterbi.image_raw)
        print("here")
        viewer.add_points([pt1], face_color='red', size=7)
        viewer.add_points([pt2], face_color='orange', size=7)
        msg = (
            f'clicked at {cords} on blob {val} which is now is displaying endpoints'
        )
    else:
        msg = f'clicked at {cords} on background which is ignored'
    print(msg)
    
napari.run()