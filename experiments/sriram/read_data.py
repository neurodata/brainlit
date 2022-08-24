import aicspylibczi
from skimage import io

path = '/cis/project/sriram/Sriram/SS IUE 175 SNOVA RFP single channel AdipoClear Brain 3 ipsilateral small z two colour Image1.czi'
czi = aicspylibczi.CziFile(path)
shape = czi.get_dims_shape()[0]
print(shape)

Z = int(shape['Z']/2)
im0 = czi.read_mosaic(C=0, Z=Z, scale_factor=1)
im1 = czi.read_mosaic(C=1, Z=Z, scale_factor=1)

outpath = "/cis/home/tathey/projects/mouselight/sriram/"
io.imsave(outpath + "im0.tif", im0)
io.imsave(outpath + "im0.tif", im1)