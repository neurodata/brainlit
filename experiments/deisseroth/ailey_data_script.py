from numpy.core.numeric import full
from cloudvolume import CloudVolume
from skimage.transform import downscale_local_mean
from skimage import io
import random
import h5py
from skimage import measure
import numpy as np 
import matplotlib.pyplot as plt 
import subprocess
from tqdm import tqdm
import pickle
from parse_ara import *
import networkx as nx
from tqdm import tqdm

path_prefix = "/data/tathey1/matt_wright/brain3/"
#path_prefix = "/Users/thomasathey/Documents/mimlab/mouselight/ailey/wholebrain_results/brain4/register/"

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_01_Sert_Cre_B/axon_mask"
vol_mask_ds = CloudVolume(dir, parallel=1, mip=1, fill_missing=True)
print(vol_mask_ds.shape)

full_data = np.zeros(vol_mask_ds.shape, dtype='uint8')
full_data = np.squeeze(full_data)
full_data = np.swapaxes(full_data, 0,2)

for i in tqdm(range(1810, vol_mask_ds.shape[2], 10)):
    z1 = i
    z2 = i+10
    if z2 >= vol_mask_ds.shape[2]:
        z2 = vol_mask_ds.shape[2]
    chunk = vol_mask_ds[:,:,z1:z2,0]
    chunk = np.squeeze(chunk)
    chunk = chunk.astype('uint8')
    chunk = 255*np.swapaxes(chunk, 0,2)
    full_data[z1:z2,:,:] = chunk
    if i%600 == 0:
        io.imsave(path_prefix + "axon_mask_1_" + str(i) + "_255.tif", full_data)

        
io.imsave(path_prefix + "axon_mask_1_255_1810-2100.tif", full_data)
