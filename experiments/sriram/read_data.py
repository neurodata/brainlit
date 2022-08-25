import aicspylibczi
from skimage import io
import numpy as np
import zarr
from tqdm import tqdm
from cloudvolume import CloudVolume

task = "writeng"

if task == "writezarr":
    sz = [2, 6814, 8448, 316]

    path = '/cis/project/sriram/Sriram/SS IUE 175 SNOVA RFP single channel AdipoClear Brain 3 ipsilateral small z two colour Image1.czi'
    czi = aicspylibczi.CziFile(path)

    print(f"Creating array of shape {sz} from czi file of shape {czi.get_dims_shape()}")

    zarra = zarr.zeros(sz, chunks=(2,100,100,40), dtype='uint16')
    num_slices = czi.get_dims_shape()[0]['Z'][1]


    for z in tqdm(np.arange(num_slices), desc="Saving slices..."):
        zarra[0,:,:,z] = np.squeeze(czi.read_mosaic(C=0, Z=z, scale_factor=1))
        zarra[1,:,:,z] = np.squeeze(czi.read_mosaic(C=1, Z=z, scale_factor=1))


    outpath = "/cis/home/tathey/projects/mouselight/sriram/"
    zarr.save(outpath + "somez.zarr", zarra)
if task == "writeng":
    outpath = "precomputed://file:///cis/home/tathey/projects/mouselight/sriram/neuroglancer_data/somez/fg"

    info = CloudVolume.create_new_info(
        num_channels    = 1,
        layer_type      = 'image',
        data_type       = 'uint16', # Channel images might be 'uint8'
        # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso
        encoding        = 'raw', 
        resolution      = [1, 1, 1], # Voxel scaling, units are in nanometers
        voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
        # Pick a convenient size for your underlying chunk representation
        # Powers of two are recommended, doesn't need to cover image exactly
        chunk_size      = [ 32, 32, 32 ], # units are voxels
        volume_size     = [ 100, 100, 100 ], # e.g. a cubic millimeter dataset
    )

    print(f"Posting info: {info}")
    vol = CloudVolume(outpath, info=info, compress = False)
    vol.commit_info()

    vol[32:64,32:64,32:64] = 256


