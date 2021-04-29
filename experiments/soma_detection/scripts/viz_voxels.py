import napari
import numpy as np
import os
from pathlib import Path

cwd = Path(os.path.abspath(__file__))
exp_dir = cwd.parents[1]
data_dir = os.path.join(exp_dir, "data")

with napari.gui_qt():
    viewer = napari.Viewer(ndisplay=3)
    vol = np.load(os.path.join(data_dir, "examples", "brain1_103.npy"))
    print(vol.shape)
    viewer.add_image(vol)
