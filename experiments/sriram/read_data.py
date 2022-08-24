import aicspylibczi
from skimage import io
import numpy as np
import zarr
from tqdm import tqdm

sz = [2, 6814, 8448, 316]

path = '/cis/project/sriram/Sriram/SS IUE 175 SNOVA RFP single channel AdipoClear Brain 3 ipsilateral small z two colour Image1.czi'
czi = aicspylibczi.CziFile(path)

print(f"Creating array of shape {sz} from czi file of shape {czi.get_dims_shape()}")

zarra = zarr.zeros(sz, chunks=(2,100,100,40), dtype='uint16')
num_slices = czi.get_dims_shape()[0]['Z'][1]

print(num_slices)

for z in tqdm(np.arange(num_slices), desc="Saving slices..."):
    zarra[0,:,:,z] = np.squeeze(czi.read_mosaic(C=0, Z=z, scale_factor=1))
    zarra[1,:,:,z] = np.squeeze(czi.read_mosaic(C=1, Z=z, scale_factor=1))


outpath = "/cis/home/tathey/projects/mouselight/sriram/"
zarr.save(outpath + "somez.zarr", zarra)