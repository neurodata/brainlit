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
    outpath_prefix = "precomputed://file:///cis/home/tathey/projects/mouselight/sriram/neuroglancer_data/somez/"

    path = '/cis/project/sriram/Sriram/SS IUE 175 SNOVA RFP single channel AdipoClear Brain 3 ipsilateral small z two colour Image1.czi'
    czi = aicspylibczi.CziFile(path)

    dims = czi.get_dims_shape()[0]

    for c, suffix in zip([0,1], ["fg", "bg"]):
        outpath = outpath_prefix + suffix
        info = CloudVolume.create_new_info(
            num_channels    = 1,
            layer_type      = 'image',
            data_type       = czi.dtype, # Channel images might be 'uint8'
            # raw, png, jpeg, compressed_segmentation, fpzip, kempressed, zfpc, compresso
            encoding        = 'raw', 
            resolution      = [1, 1, 1], # Voxel scaling, units are in nanometers
            voxel_offset    = [0, 0, 0], # x,y,z offset in voxels from the origin
            # Pick a convenient size for your underlying chunk representation
            # Powers of two are recommended, doesn't need to cover image exactly
            chunk_size      = [ 128, 128, 1 ], # units are voxels
            volume_size     = [ dims['X'][1], dims['Y'][1], dims['Z'][1]], # e.g. a cubic millimeter dataset
        )

        print(f"Posting info: {info}")
        vol = CloudVolume(outpath, info=info, compress = False)
        vol.commit_info()

        num_slices = dims['Z'][1]
        for z in tqdm(np.arange(num_slices), desc="Saving slices..."):
            im_slice = np.squeeze(czi.read_mosaic(C=0, Z=z, scale_factor=1))
            im_slice = np.expand_dims(im_slice, axis=2)
            vol[:,:,z] = im_slice


