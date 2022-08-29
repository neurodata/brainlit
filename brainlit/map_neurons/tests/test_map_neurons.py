from pathlib import Path
import os
import numpy as np
import h5py
from scipy.io import savemat

root_dir = Path(os.path.abspath(""))
data_dir = os.path.join(root_dir, "data", "testing")
h5_path = data_dir + "A.h5"


