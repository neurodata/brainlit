from cloudvolume import CloudVolume
from tqdm import tqdm

for i in range(4):
    vol = CloudVolume(
        "precomputed://s3://smartspim-precomputed-volumes/2022_10_26/11537/Ch_561",
        mip=i,
    )
    for z in range(4599, 4799):
        vol[:, :, z, :] = 7
