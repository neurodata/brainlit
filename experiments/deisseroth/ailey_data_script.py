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

dir = "precomputed://https://dlab-colm.neurodata.io/2021_07_15_Sert_Cre_R/axon_mask"
vol_mask_ds = CloudVolume(dir, parallel=1, mip=1, fill_missing=False)
print(vol_mask_ds.shape)

data = vol_mask_ds[:,:,:,:]
data = data[:,:,:,0]
data = np.swapaxes(data, 0,2)

io.imsave("/data/tathey1/matt_wright/brain4/axon_mask_1.tif", data)
