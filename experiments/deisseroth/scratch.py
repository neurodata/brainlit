from cloudvolume import CloudVolume
import numpy as np

path = "file:///mnt/data/Neuroglancer_data/2021_10_06/8557/Ch_647"
vol = CloudVolume(path, parallel=1, mip=0, fill_missing=False)

im = vol[2500:2540, 4900:4940, 3000:3040]

print(f"647 sum: {np.sum(im)}")

path = "file:///mnt/data/Neuroglancer_data/2021_10_06/8557/Ch_561"
vol = CloudVolume(path, parallel=1, mip=0, fill_missing=False)

im = vol[2500:2540, 4900:4940, 3000:3040]

print(f"647 sum: {np.sum(im)}")