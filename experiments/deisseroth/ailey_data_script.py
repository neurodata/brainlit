from cloudvolume import CloudVolume
from skimage.transform import downscale_local_mean
import napari
from skimage import io
import random
import h5py
from skimage import measure
from brainlit.preprocessing import removeSmallCCs
import numpy as np 
import matplotlib.pyplot as plt 
import subprocess
import tables
from napari_animation import AnimationWidget
from tqdm import tqdm
import pickle
from parse_ara import *
import networkx as nx
import seaborn as sns
import pandas as pd
import brainrender
import scipy.ndimage as ndi
from skimage.morphology import skeletonize
%gui qt5

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/axon_mask"
vol_mask_ds = CloudVolume(dir, parallel=1, mip=2, fill_missing=False)
print(vol_mask_ds.shape)

data = vol_mask_ds[:,:,:,:]
data = data[:,:,:,0]
data = np.swapaxes(data, 0,2)

io.imsave("/data/tathey1/matt_wright/brain4/axon_mask_2.tif", data)