import napari
import zarr
import pickle
import os
from pathlib import Path

scale = [0.3, 0.3, 1]


root_dir = Path(os.path.abspath(""))
data_dir = os.path.join(root_dir, "data", "example")
path = os.path.join(data_dir, "3-1-soma_viterbrain.pickle")

with open(path, "rb") as handle:
    viterbi = pickle.load(handle)

im = zarr.open('/Users/sejalsrivastava/brainlit/experiments/ViterBrain/data/example/3-1-soma.zarr', "r")
labels = zarr.open('/Users/sejalsrivastava/brainlit/experiments/ViterBrain/data/example/3-1-soma_labels.zarr', "r")
viewer = napari.Viewer(ndisplay=3)
viewer.add_image(im, name="image", scale=scale)
labels_layer = viewer.add_labels(labels, name="labels", scale=scale)
napari.run()