# -*- coding: utf-8 -*-
"""
Created on Sun Nov 29 15:47:49 2020

@author: frede
"""
import numpy as np
import pandas as pd
import napari
from pyVaa3d.vaa3dWrapper import runVaa3dPlugin
from skimage import io
import os
from mouselight_code.src.benchmarking_params import brain_offsets, vol_offsets, scales, type_to_date
from mouselight_code.src import read_swc
from brainlit.utils.swc import df_to_graph, graph_to_paths
from pathlib import Path
from sklearn.metrics import pairwise_distances_argmin_min

# For unpacking the APP2 output .swc files as a list of labels
def app2_swc_unpack(fname):
    """Reads in the results swc file and converts it to a list of (x,y,z)
       coordinates corresponding to labels
    """

    # check input
    file = open(fname, "r")
    skip_header = True
    readable = True
    header_length = -1
    while skip_header:
        line = file.readline().split()
        if line != []:
            if line[0][0] != "#":
                #line = file.readline().split()
                skip_header = False
        else:
            readable = False
            skip_header = False
            
        header_length += 1
    # read coordinates
    if readable:
        df = pd.read_table(
            fname,
            names=["n", "type", "x", "y", "z", "radius", "parent"],
            skiprows=header_length,
            sep="\s"
        )
    else:
        return None
     
    # NOTE: The read-in order is [z,y,x] since the image is flipped.
    labels = np.array(df[["z","y","x"]])
    return labels

def ssd(pts1, pts2):
    """Compute significant spatial distance metric between two traces as defined in APP1.
    Args:
        pts1 (np.array): array containing coordinates of points of trace 1. shape: npoints x ndims
        pts2 (np.array): array containing coordinates of points of trace 1. shape: npoints x ndims
    Returns:
        [float]: significant spatial distance as defined by APP1
    """
    _, dists1 = pairwise_distances_argmin_min(pts1, pts2)
    dists1 = dists1[dists1 >= 2]
    _, dists2 = pairwise_distances_argmin_min(pts2, pts1)
    dists2 = dists2[dists2 >= 2]
    dists = np.concatenate([dists1, dists2])
    ssd = np.mean(dists)
    return ssd

#%%
directory = r'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\'
for n in range(1,26):
    fname = 'test_' + str(n) + '-gfp.tif'
    img_name = os.path.join(directory, fname)
    img = io.imread(img_name)

    # NOTE: This will autodump files into the dataset folder, since app2
    # is set up to dump in the source folder.
    # I move them manually to an app2-output folder.
    print(f"Running APP2 on image {fname}.")
    runVaa3dPlugin(inFile=img_name, pluginName="Vaa3D_Neuron2",
                   funcName="app2")

print(f"APP2 Done. Please check {directory} for outputs.")

#%%
directory = r'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\'

test_results = {}

for f in os.listdir(directory):
    if f.endswith("app2.swc"):
        test_results[f.split(sep='.')[0]] = app2_swc_unpack(os.path.join(directory, f))

#%%
'''Note: This is just launcher code for napari so I can qualitatively analyze
each of these files to see if they ran well. If they ran well, I can further
analyze their distance metrics.'''

raw_directory = r'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\'

fnames = list(test_results.keys())
select_file = fnames[7]
print(select_file)
img_name = select_file+".tif"

img = io.imread(os.path.join(raw_directory, img_name))
img_labels = test_results[select_file]
print(img_labels)

with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_points(img_labels, size=1, opacity=0.5, face_color='yellow',edge_color='red')
    
#%%
target_files = [fnames[6], fnames[7], fnames[13]] #[1,6,7,13]

im_dir = Path(
    "C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\"
)
gfp_files = list(im_dir.glob("**/*-gfp.tif"))
swc_base_path = im_dir / "Manual-GT"

gt_labels = {}

# for every image
for im_num, im_path in enumerate(gfp_files):
    
    f = im_path.parts[-1][:-8].split("_")
    image = f[0]

    date = type_to_date[image]
    num = int(f[1])
    img_name = image + "_" + str(num) + "-gfp"
    
    # Skip the file if it's not of interest
    if img_name not in target_files:
        continue
    
    # More file reading
    scale = scales[date]
    brain_offset = brain_offsets[date]

    vol_offset = vol_offsets[date][num]
    im_offset = np.add(brain_offset, vol_offset)
    
    # Parse the proper directory
    lower = int(np.floor((num - 1) / 5) * 5 + 1)
    upper = int(np.floor((num - 1) / 5) * 5 + 5)
    dir1 = date + "_" + image + "_" + str(lower) + "-" + str(upper)
    dir2 = date + "_" + image + "_" + str(num)
    swc_path = swc_base_path / dir1 / dir2
    
    # Read swcs
    swc_files = list(swc_path.glob("**/*.swc"))
    #im = io.imread(im_path, plugin="tifffile")
    #print(f"Image shape: {im.shape}")
    path_total = []
    for swc_num, swc in enumerate(swc_files):
        if "cube" in swc.parts[-1]:
            # skip the bounding box swc
            continue

        df, swc_offset, _, _, _ = read_swc.read_swc(swc)

        #compute the offset of the swc relative to the image
        offset_diff = np.subtract(swc_offset, im_offset)
        G = df_to_graph(df)

        paths = graph_to_paths(G)
        # for every path in that swc
        for path_num, p in enumerate(paths):
            #convert from spatial coordinates to voxel coordinates
            pvox = (p + offset_diff) / (scale) * 1000
            
            # Swap the columns to orient properly
            pvox[:, [2, 0]] = pvox[:, [0, 2]] 
            path_total.extend(pvox)

    gt_labels[img_name] = path_total



#%% Compute SSD
SSD = {}
for img_name in list(gt_labels.keys()):
    SSD[img_name] = ssd(test_results[img_name], gt_labels[img_name])

    img = io.imread(os.path.join(raw_directory, img_name + ".tif"))
    with napari.gui_qt():
        viewer = napari.Viewer(ndisplay=3)
        viewer.add_image(img)
        viewer.add_points(gt_labels[img_name], size=0.8, opacity=0.5, face_color='green',edge_color='green')
        viewer.add_points(test_results[img_name], size=0.8, opacity=0.5, face_color='yellow',edge_color='red')     
print(SSD)