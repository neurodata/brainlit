import aicspylibczi
from skimage import io
import numpy as np
import zarr

path = '/cis/project/sriram/Sriram/SS IUE 175 SNOVA RFP single channel AdipoClear Brain 3 ipsilateral small z two colour Image1.czi'
czi = aicspylibczi.CziFile(path)
sz = np.squeeze(czi.size)
print(sz)
raise ValueError()

zarra = zarr.zeros(sz, chunks=(2,100,100,40), dtype='uint16')
for z in range(czi.get_dims_shape()[0]['Z'][0]):
    zarra[0,:,:,z] = np.squeeze(czi.read_mosaic(C=0, Z=Z, scale_factor=1))
    zarra[1,:,:,z] = np.squeeze(czi.read_mosaic(C=1, Z=Z, scale_factor=1))


outpath = "/cis/home/tathey/projects/mouselight/sriram/"
zarr.save(outpath + "somez.zarr", zarra)