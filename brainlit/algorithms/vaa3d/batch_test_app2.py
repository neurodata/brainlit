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
from brainlit.utils.swc import read_swc
from brainlit.utils.swc import read_swc_offset
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

'''
# For unpacking the ground truth .swc files as a list of labels
def GT_swc_unpack(fname):
    file = open(fname, "r")
    skip_header = True
    readable = True
    header_length = -1
    while skip_header:
        line = file.readline().split()
        if line != []:
            if line[0][0] != "#":
                skip_header = False
                
        header_length += 1   
    
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
    labels = np.array(df[["x","y","z"]])
    return labels
'''
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
select_file = fnames[6]
print(select_file)
img_name = select_file+".tif"

img = io.imread(os.path.join(raw_directory, img_name))
img_labels = test_results[select_file]
print(img_labels)

with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_points(img_labels, size=0.5, opacity=0.5, face_color='yellow',edge_color='red')
    
#%%
good_file_inds = [6] #[1,6,7,13]
for n in good_file_inds:
    fname = fnames[n]
    
    # Parse the file number
    fnum = int(fname.split(sep='_')[1].split(sep='-')[0])
    
    # Browse the proper directory for the ground truth .swc files
    if fnum >= 1 and fnum <= 5:
        GT_directory = 'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\Manual-GT\\8-01_test_1-5\\8-01_test_'+str(fnum)
    elif fnum >= 6 and fnum <= 10:
        GT_directory = 'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\Manual-GT\\8-01_test_6-10\\8-01_test_'+str(fnum)
    elif fnum >= 11 and fnum <= 15:
        GT_directory = 'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\Manual-GT\\8-01_test_11-15\\8-01_test_'+str(fnum)
    elif fnum >= 16 and fnum <= 20:
        GT_directory = 'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\Manual-GT\\8-01_test_16-20\\8-01_test_'+str(fnum)
    elif fnum >= 21 and fnum <= 25:
        GT_directory = 'C:\\Users\\frede\\Documents\\Y4\\Y4_NDD\\benchmarking_datasets\\Manual-GT\\8-01_test_21-25\\8-01_test_'+str(fnum)
    else:
        print("Invalid file.")
        continue
    gt_labels = []
    for gt_file in os.listdir(GT_directory):
        if gt_file.endswith('.swc'):
            labels_df,offset,_,_,_ = read_swc(os.path.join(GT_directory, gt_file))
            #labels_df,_,_,_ = read_swc_offset(os.path.join(GT_directory, gt_file))
            
            labels_list = np.array(labels_df[['z','y','x']])
            gt_labels.extend(labels_list)

select_file = fnames[6]
print(select_file)
img_name = select_file+".tif"
img = io.imread(os.path.join(raw_directory, img_name))
with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img)
    viewer.add_points(gt_labels, size=0.5, opacity=0.5, face_color='yellow',edge_color='red')