# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 012:11:28 2020

@author: frede
"""

from brainlit.utils.session import NeuroglancerSession
from brainlit.utils.swc import graph_to_paths
import numpy as np
import pandas as pd
import napari
import matplotlib.pyplot as plt
from pyVaa3d.vaa3dWrapper import runVaa3dPlugin
from skimage import io


def plot_labels(labels, image_shape, name):
    labels_ind = np.nonzero(labels)

    fig = plt.figure()    
    ax = fig.gca(projection='3d')
    ax.scatter(labels_ind[0], labels_ind[1], labels_ind[2],c="blue",s=0.2)
    plt.title(name)     
    ax.set_xlim3d(0, image_shape[0])
    ax.set_ylim3d(0, image_shape[1])
    ax.set_zlim3d(0, image_shape[2])

def swc_unpack(fname):
    """Reads in the results swc file and converts it to a list of (x,y,z)
       coordinates corresponding to labels
    """

    # check input
    file = open(fname, "r")
    skip_header = True
    header_length = -1
    while skip_header:
        line = file.readline().split()
        if line[0][0] != "#":
            #line = file.readline().split()
            skip_header = False
        header_length += 1    
    print(header_length)
    # read coordinates
    df = pd.read_table(
        fname,
        names=["n", "type", "x", "y", "z", "radius", "parent"],
        skiprows=header_length,
        sep="\s"
    )
      
    labels = np.array(df[["x","y","z"]])
    return labels
    
def fix_labels(labels):
    """ For some reason, the labels are flipped. This performs a flip that
    will fix it. 
    """

#%% Grab volume from S3
dir = "s3://open-neurodata/brainlit/brain1"
dir_segments = "s3://open-neurodata/brainlit/brain1_segments"
mip = 0
v_id = 0
radius = 75

# get image and center point
ngl_sess = NeuroglancerSession(mip = mip, url = dir, url_segments=dir_segments)
img, bbox, vox = ngl_sess.pull_voxel(2, v_id, radius)
print(f"\n\nDownloaded volume is of shape {img.shape}, with total intensity {sum(sum(sum(img)))}.")

#%% Save as .tif
fname = "brain1_demo_segments.tif"
io.imsave(fname,img)
print(f"\n\nImage {fname} saved.")

#%%

# Change the path name to match a local file location
img_name = "C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\brain1_demo_segments.tif"

# Check the README.md for procedure to install the Vaa3d plugin
runVaa3dPlugin(inFile=img_name, pluginName="Vaa3D_Neuron2",
                   funcName="app2")

print("Algorithm successfully run! Please check the root folder for the .swc output file.")

#%%

# Change the path name to match a local file location
swc_name = "C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\brain1_demo_segments.tif_ini.swc"
labels = swc_unpack(swc_name)
print(labels)
print(f"\n\n Labels successfully loaded from .swc file with {len(labels)} foreground labels.")

#%%
G_sub = ngl_sess.get_segments(2, bbox)
paths = graph_to_paths(G_sub)
print(f"Selected volume contains {G_sub.number_of_nodes()} nodes and {len(paths)} paths")

with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_shapes(data=paths, shape_type='path', edge_width=0.1, edge_color='blue', opacity=0.1)
    viewer.add_points(vox, size=1, opacity=0.5)

    viewer.add_points(labels, size=0.5, opacity=0.5, face_color='red',edge_color='yellow')
