from cloudvolume import CloudVolume

dir = "s3://open-neurodata/brainlit/brain1"

vol = CloudVolume(dir, parallel=1, mip=0, fill_missing=True)

print("downloading")
img = vol[11210:14542,10811:14143,3977:4977]

print("saving")
io.imsave("/data/tathey1/mouselight/1mm.tif", img)

