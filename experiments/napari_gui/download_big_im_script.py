from cloudvolume import CloudVolume
from skimage import io 

dir = "s3://open-neurodata/brainlit/brain1"

vol = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)
print(f"shape{vol.shape}")

print("downloading")
img = vol[11210:14542,10811:14143,3977:4977]

print("saving")
io.imsave("/data/tathey1/mouselight/1mm.tif", img)

